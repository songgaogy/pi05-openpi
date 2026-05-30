export CLIENT_ARGS="--args.task-suite-name libero_10 --args.video-out-path /data/libero/videos"

SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir /app/checkpoints/pi05_libero" \
docker compose -f examples/libero/compose.yml up --build