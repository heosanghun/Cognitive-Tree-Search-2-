# Cognitive Tree Search (CTS) Project Agent Rules

## GPU Resource Allocation Constraints
- **CRITICAL**: SSH 원격 서버 접속 및 연산 수행 시, 사용할 GPU 장치는 반드시 **0번 ~ 3번 (GPU 0 ~ GPU 3, cuda:0 ~ cuda:3)** 범위 내에서만 할당해야 합니다.
- **GPU 4번 이상 (cuda:4 등)의 장치는 연구실의 허가를 받지 않은 장치이며 규정 위반이 되므로 절대 지정하거나 사용해서는 안 됩니다.**
- 스크립트 실행 파라미터(예: `--device cuda:X`) 지정 시 반드시 `cuda:0`, `cuda:1`, `cuda:2`, `cuda:3` 중 하나를 할당하도록 장치 코드를 명시적으로 설정하십시오.
