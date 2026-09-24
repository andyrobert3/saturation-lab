//! Mock target with a known capacity: N concurrent requests, each holding a slot for D seconds. Arithmetic: capacity = N/D req/s.

use axum::{
    Json, Router,
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
};
use futures_util::{Stream, stream};
use serde_json::{Value, json};
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::{
    sync::Semaphore,
    time::{Instant, sleep_until},
};

struct AppState {
    slots: Arc<Semaphore>,
    service_time: Duration,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    _body: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Queueing, not rejecting: overloaded requests wait, so served rate caps at N/D while latency grows — the knee.
    let permit = state.slots.clone().acquire_owned().await.unwrap();
    let stream_start = Instant::now();
    let output = stream::unfold(
        (permit, state, stream_start, 0usize),
        |(permit, state, stream_start, index)| async move {
            let event = match index {
                0..16 => {
                    sleep_until(
                        stream_start + state.service_time.mul_f64((index + 1) as f64 / 16.0),
                    )
                    .await;
                    Event::default().data(
                        json!({
                            "id": "mock",
                            "choices": [{
                                "text": "m",
                                "index": 0,
                                "token_ids": [index],
                                "finish_reason": if index == 15 { Some("length") } else { None }
                            }]
                        })
                        .to_string(),
                    )
                }
                16 => Event::default().data(
                    json!({
                        "id": "mock",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 16,
                            "total_tokens": 17
                        }
                    })
                    .to_string(),
                ),
                17 => Event::default().data("[DONE]"),
                _ => return None,
            };
            Some((Ok(event), (permit, state, stream_start, index + 1)))
        },
    );
    Sse::new(output)
}

async fn version() -> Json<Value> {
    Json(json!({ "version": "mock-0.1.0" }))
}

fn die(msg: &str) -> ! {
    eprintln!("mock: {msg}");
    std::process::exit(2);
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        die("usage: mock <n> <d> [port]");
    }
    let n: usize = args[1]
        .parse()
        .unwrap_or_else(|_| die("n must be a positive integer"));
    let d: f64 = args[2]
        .parse()
        .unwrap_or_else(|_| die("d must be a number of seconds"));
    if n == 0 || d <= 0.0 {
        die("n must be > 0 and d must be > 0");
    }
    let port: u16 = args
        .get(3)
        .map(|s| s.parse().unwrap_or_else(|_| die("port must be 0-65535")))
        .unwrap_or(8080);

    let state = Arc::new(AppState {
        slots: Arc::new(Semaphore::new(n)),
        service_time: Duration::from_secs_f64(d),
    });

    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/version", get(version))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .unwrap_or_else(|e| die(&format!("cannot bind port {port}: {e}")));
    let addr = listener.local_addr().unwrap();
    println!(
        "mock: capacity = {n}/{d} = {:.1} req/s, listening on {addr}",
        n as f64 / d
    );
    axum::serve(listener, app).await.unwrap();
}
