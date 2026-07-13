# Agent Instructions

## GPU Usage Constraints
- **GPU 사용 지침 (GPU Usage Constraint)**:
  - SSH 서버에 접속하여 연산 및 학습을 수행할 때는 **GPU 0번, 1번, 2번, 3번까지만** 사용해야 합니다.
  - 배포 스크립트 또는 훈련 스크립트(예: docker run, CUDA_VISIBLE_DEVICES, device 지정 등) 작성 및 수정 시, **GPU 4번, 5번, 6번, 7번(또는 그 이상)은 절대 할당하거나 침범하지 않도록** 설정을 설계하고 즉시 수정하십시오.
  - When accessing the SSH server, agents must restrict VRAM and process allocation to **GPUs 0, 1, 2, and 3 only**.
  - All deployment and run scripts (e.g., `docker run --gpus`, `CUDA_VISIBLE_DEVICES`, `--device`) must be immediately updated to use only GPUs 0-3. Never configure or intrude on GPUs 4-7.
