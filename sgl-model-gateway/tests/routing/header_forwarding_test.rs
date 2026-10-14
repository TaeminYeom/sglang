//! Header forwarding integration tests
//!
//! Tests for header propagation through the router to workers.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

#[cfg(test)]
mod worker_auth_tests {
    use std::{sync::Arc, time::Duration};

    use axum::{
        body::{to_bytes, Body},
        http::{header::AUTHORIZATION, HeaderMap, Request, StatusCode},
        Router as AxumRouter,
    };
    use smg::{
        config::RouterConfig,
        core::BasicWorkerBuilder,
        protocols::generate::GenerateRequest,
        routers::{http::router::Router, RouterTrait},
    };
    use tokio::{
        net::TcpListener,
        sync::{mpsc, Barrier},
        task::JoinHandle,
        time::timeout,
    };

    use crate::common::create_test_context;

    struct AuthWorker {
        url: String,
        requests: mpsc::UnboundedReceiver<HeaderMap>,
        server: JoinHandle<()>,
    }

    impl AuthWorker {
        async fn start(barrier: Option<Arc<Barrier>>) -> Self {
            let (sender, requests) = mpsc::unbounded_channel();
            let app = AxumRouter::new().fallback(move |headers: HeaderMap| {
                let sender = sender.clone();
                let barrier = barrier.clone();
                async move {
                    sender.send(headers).unwrap();
                    // Broadcast returns on the first successful response. Wait for
                    // every worker to receive its request before allowing responses.
                    if let Some(barrier) = barrier {
                        barrier.wait().await;
                    }
                    "data: [DONE]\n\n"
                }
            });
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Self {
                url,
                requests,
                server,
            }
        }

        async fn assert_auth(&mut self, expected: Option<&str>) {
            let headers = timeout(Duration::from_secs(5), self.requests.recv())
                .await
                .expect("Worker did not receive a request")
                .unwrap();
            let actual: Vec<_> = headers
                .get_all(AUTHORIZATION)
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect();
            let expected: Vec<_> = expected.into_iter().collect();
            assert_eq!(actual, expected, "Unexpected worker authorization");
        }
    }

    impl Drop for AuthWorker {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    async fn create_router(workers: &[(&AuthWorker, Option<&str>)]) -> Router {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .api_key("router-key")
            .build_unchecked();
        let context = create_test_context(config).await;
        for (worker, key) in workers {
            let mut builder = BasicWorkerBuilder::new(&worker.url);
            if let Some(key) = key {
                builder = builder.api_key(*key);
            }
            context.worker_registry.register(Arc::new(builder.build()));
        }
        Router::new(&context).await.unwrap()
    }

    fn client_headers(auth: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(auth) = auth {
            headers.insert(AUTHORIZATION, auth.parse().unwrap());
        }
        headers
    }

    const AUTH_CASES: [(Option<&str>, Option<&str>, Option<&str>); 4] = [
        (
            Some("worker-key"),
            Some("Bearer router-key"),
            Some("Bearer worker-key"),
        ),
        (Some("worker-key"), None, Some("Bearer worker-key")),
        (None, Some("Bearer client-key"), Some("Bearer client-key")),
        (None, None, None),
    ];

    /// Client credentials must not accompany the worker's token, including
    /// streaming requests; keyless workers retain authorization passthrough.
    #[tokio::test]
    async fn test_inference_worker_auth_priority() {
        for (key, client_auth, expected) in AUTH_CASES {
            let mut worker = AuthWorker::start(None).await;
            let router = create_router(&[(&worker, key)]).await;
            let headers = client_headers(client_auth);
            for stream in [false, true] {
                let request: GenerateRequest = serde_json::from_value(serde_json::json!({
                    "text": "hello", "stream": stream,
                }))
                .unwrap();
                let response = router.route_generate(Some(&headers), &request, None).await;
                assert_eq!(response.status(), StatusCode::OK);
                to_bytes(response.into_body(), usize::MAX).await.unwrap();
                worker.assert_auth(expected).await;
            }
        }
    }

    /// GET proxies must use registered worker credentials rather than relying
    /// on clients to know the selected worker's key.
    #[tokio::test]
    async fn test_proxy_get_worker_auth_priority() {
        for (key, client_auth, expected) in AUTH_CASES {
            let mut worker = AuthWorker::start(None).await;
            let router = create_router(&[(&worker, key)]).await;
            let mut request = Request::new(Body::empty());
            *request.headers_mut() = client_headers(client_auth);
            let response = router.get_models(request).await;
            assert_eq!(response.status(), StatusCode::OK);
            worker.assert_auth(expected).await;
        }
    }

    /// A broadcast must authenticate independently to differently keyed workers
    /// while preserving client authorization for a keyless worker.
    #[tokio::test]
    async fn test_broadcast_uses_each_workers_auth() {
        for client_auth in [Some("Bearer router-key"), None] {
            let barrier = Arc::new(Barrier::new(3));
            let mut worker_a = AuthWorker::start(Some(barrier.clone())).await;
            let mut worker_b = AuthWorker::start(Some(barrier.clone())).await;
            let mut keyless_worker = AuthWorker::start(Some(barrier)).await;
            let router = create_router(&[
                (&worker_a, Some("worker-a-key")),
                (&worker_b, Some("worker-b-key")),
                (&keyless_worker, None),
            ])
            .await;
            let headers = client_headers(client_auth);
            let response = timeout(
                Duration::from_secs(5),
                router.cancel_response(Some(&headers), "response-id"),
            )
            .await
            .expect("Broadcast did not reach every worker");
            assert_eq!(response.status(), StatusCode::OK);
            worker_a.assert_auth(Some("Bearer worker-a-key")).await;
            worker_b.assert_auth(Some("Bearer worker-b-key")).await;
            keyless_worker.assert_auth(client_auth).await;
        }
    }
}

#[cfg(test)]
mod header_forwarding_tests {
    use super::*;

    /// Test that X-Request-Id header is forwarded
    #[tokio::test]
    async fn test_request_id_forwarding() {
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port: 19400,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let custom_request_id = "test-request-id-12345";
        let payload = json!({
            "text": "Test header forwarding",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-request-id", custom_request_id)
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Response should have the same request ID
        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some(), "Response should have x-request-id");
        assert_eq!(
            response_id.unwrap().to_str().unwrap(),
            custom_request_id,
            "Request ID should be preserved"
        );

        ctx.shutdown().await;
    }

    /// Test custom request ID headers
    #[tokio::test]
    async fn test_custom_request_id_headers() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3500)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .request_id_headers(vec![
                "custom-trace-id".to_string(),
                "x-correlation-id".to_string(),
            ])
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 19401,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        let custom_trace_id = "my-custom-trace-123";
        let payload = json!({
            "text": "Test custom headers",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("custom-trace-id", custom_trace_id)
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Response should use the custom header as request ID
        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());
        assert_eq!(
            response_id.unwrap().to_str().unwrap(),
            custom_trace_id,
            "Custom header should be used as request ID"
        );

        ctx.shutdown().await;
    }

    /// Test correlation ID header forwarding
    #[tokio::test]
    async fn test_correlation_id_forwarding() {
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port: 19402,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let correlation_id = "correlation-abc-789";
        let payload = json!({
            "text": "Test correlation ID",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-correlation-id", correlation_id)
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Response should preserve the correlation ID as request ID
        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());
        assert_eq!(
            response_id.unwrap().to_str().unwrap(),
            correlation_id,
            "Correlation ID should be preserved"
        );

        ctx.shutdown().await;
    }

    /// Test that request ID is generated when not provided
    #[tokio::test]
    async fn test_auto_generated_request_id() {
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port: 19403,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Test auto-generated ID",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Response should have an auto-generated request ID
        let response_id = resp.headers().get("x-request-id");
        assert!(
            response_id.is_some(),
            "Response should have auto-generated x-request-id"
        );

        let id_value = response_id.unwrap().to_str().unwrap();
        assert!(!id_value.is_empty(), "Request ID should not be empty");
        // For generate endpoint, ID should have 'gnt-' prefix
        assert!(
            id_value.starts_with("gnt-"),
            "Generate endpoint should have gnt- prefix, got: {}",
            id_value
        );

        ctx.shutdown().await;
    }

    /// Test chat completions request ID format
    #[tokio::test]
    async fn test_chat_completions_request_id_format() {
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port: 19404,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Response should have chatcmpl- prefix for chat completions
        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());

        let id_value = response_id.unwrap().to_str().unwrap();
        assert!(
            id_value.starts_with("chatcmpl-"),
            "Chat completions should have chatcmpl- prefix, got: {}",
            id_value
        );

        ctx.shutdown().await;
    }

    /// Test multiple header priorities
    #[tokio::test]
    async fn test_header_priority() {
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port: 19405,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let primary_id = "primary-request-id";
        let fallback_id = "fallback-correlation-id";
        let payload = json!({
            "text": "Test header priority",
            "stream": false
        });

        // When x-request-id is provided, it should take priority
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-request-id", primary_id)
            .header("x-correlation-id", fallback_id)
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());
        assert_eq!(
            response_id.unwrap().to_str().unwrap(),
            primary_id,
            "x-request-id should take priority"
        );

        ctx.shutdown().await;
    }
}
