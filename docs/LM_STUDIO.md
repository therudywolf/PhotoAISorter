# LM Studio Setup

Photo AI Sorter talks to LM Studio through the OpenAI-compatible API.

## Recommended Server Settings

- Enable the local server in LM Studio.
- Use a vision or multimodal model for sorting and duplicate verification.
- Keep the API base in the app pointed at the server root, for example `http://127.0.0.1:1234`.
- Leave the API key empty unless your server explicitly requires bearer auth.
- If auth is required, set `PHOTO_AI_SORTER_API_KEY` in the environment before launching the app.
- Optional defaults can be overridden with `PHOTO_AI_SORTER_API_BASE` and `PHOTO_AI_SORTER_MODEL`.

If your LM Studio server uses authentication and runs on a different host, use:

```bash
PHOTO_AI_SORTER_API_BASE=http://your-server:port
PHOTO_AI_SORTER_API_KEY=your-lm-studio-token
```

Put those values in `.env.local` or paste the token into the GUI `API key` field.
Do not commit `.env.local`; it is intentionally ignored.

The app uses LM Studio's OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

If you paste `http://host:port/v1` or `http://host:port/api/v1`, the app normalizes
it back to the server root before building requests.

## Model Selection

Use `Refresh models`, select a candidate, then run `Vision self-test`. For sorting large libraries, run `Benchmark` on several visible models and save the best result into the classifier profile.

Useful profile defaults:

- Classifier: vision-capable, low temperature, short output.
- Duplicate verifier: vision-capable, slightly longer output, low temperature.
- Fast preview: smaller model when available.

## Large Library Safety

For the first pass over a large mixed library:

- Use `Review-first` so the app writes manifests without copying files.
- Prefer `Preset: my tags` or `Smart auto categories`.
- Avoid `Model tags` until you have tested a small sample.
- Keep aliases updated when the model creates near-duplicate names.

## Troubleshooting

- `Vision self-test` fails: verify that the model supports image inputs.
- `/v1/models` is empty: confirm that the LM Studio server is running and the base URL is correct.
- `401 Unauthorized`: LM Studio authentication is enabled; paste the active API token into the app or set `PHOTO_AI_SORTER_API_KEY`.
- Requests time out: reduce workers in the profile or choose a smaller model.
- Too many folders appear: add aliases and re-run with `Smart auto categories`.
