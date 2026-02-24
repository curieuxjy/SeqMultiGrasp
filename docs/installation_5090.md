# Installation Guide for RTX 5090

RTX 5090(Blackwell, compute capability 12.0) 전용 설치 가이드입니다.

## 사전 요구사항

- **NVIDIA 드라이버**: 570.x 이상 (확인: `nvidia-smi`)
- **CUDA Toolkit**: 12.8 이상 (확인: `nvcc --version`)
- **OS**: Ubuntu 22.04/24.04 또는 Debian 13

```bash
# 확인
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
# 예상 출력: NVIDIA GeForce RTX 5090, 580.xx.xx, 12.0

nvcc --version
# 예상 출력: release 12.8 이상
```

## Step 1: Conda 환경 생성

Python 3.10을 권장합니다 (PyTorch 2.7+ 호환성 최적).

```bash
conda create -n seq_multi_grasp python=3.10
conda activate seq_multi_grasp
```

## Step 2: PyTorch 설치 (CUDA 12.8 빌드)

RTX 5090은 `cu128` 빌드가 필요합니다. `cu118`이나 `cu124` 빌드에는 sm_120 커널이 포함되어 있지 않아 동작하지 않습니다.

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge libstdcxx-ng
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print('sm_120 지원:', 'sm_120' in str(torch.cuda.get_arch_list()))"
```

## Step 3: TORCH_CUDA_ARCH_LIST 환경변수 설정

RTX 5090(sm_120)용 CUDA 커널을 컴파일하기 위해 **모든 후속 빌드 전에** 이 환경변수를 설정해야 합니다. 이것을 빠뜨리면 `RuntimeError: CUDA error: no kernel image is available for execution on the device` 에러가 발생합니다.

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

> **참고**: `.bashrc`에 추가하거나, 매 터미널 세션마다 설정하세요.

## Step 4: PyTorch3D 빌드

사전 빌드된 PyTorch3D wheel에는 sm_120이 포함되어 있지 않으므로 소스에서 빌드해야 합니다.

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

빌드에 시간이 오래 걸릴 수 있습니다 (10~30분). 빌드 실패 시 `MAX_JOBS`를 줄여보세요:

```bash
MAX_JOBS=4 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Step 5: ManiSkill3 설치

```bash
pip install git+https://github.com/haosulab/ManiSkill.git@v3.0.0b21
```

## Step 6: 기타 의존성 설치

```bash
pip install -r requirements.txt
```

## Step 7: 서브모듈 초기화

```bash
git submodule update --init --recursive
```

## Step 8: pointnet2_ops_lib 설치 (패치 필요)

`third-party/pointnet2_ops_lib/setup.py`에 RTX 5090 아키텍처(12.0)가 하드코딩되어 있지 않습니다. 빌드 전에 수정이 필요합니다.

```bash
# setup.py의 TORCH_CUDA_ARCH_LIST 라인을 패치
sed -i 's/os.environ\["TORCH_CUDA_ARCH_LIST"\] = "5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"/os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"/' third-party/pointnet2_ops_lib/setup.py

# 빌드
cd third-party/pointnet2_ops_lib
python setup.py develop
cd ../..
```

또는 환경변수를 직접 오버라이드해도 됩니다 (setup.py가 덮어쓰므로 패치가 더 확실함):

```bash
# 대안: setup.py 수정 없이 직접 환경변수로 강제
cd third-party/pointnet2_ops_lib
TORCH_CUDA_ARCH_LIST="12.0" python setup.py develop
cd ../..
```

> **주의**: `setup.py` 19번째 줄에서 `os.environ["TORCH_CUDA_ARCH_LIST"]`를 직접 덮어쓰기 때문에, 환경변수만 설정하는 것으로는 부족합니다. 반드시 setup.py를 수정하거나, sed 패치를 적용하세요.

## Step 9: CuRobo 설치

CuRobo는 `TORCH_CUDA_ARCH_LIST`를 하드코딩하지 않으므로 환경변수만 설정하면 됩니다.

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
cd third-party/curobo
pip install -e . --no-build-isolation
cd ../..
```

## Step 10: Kaolin 설치

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export IGNORE_TORCH_VER=1
cd third-party/kaolin
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
pip install -e .
cd ../..
```

## Step 11: Allegro Visualization 설치

순수 Python 패키지로 CUDA 빌드 불필요합니다.

```bash
cd third-party/allegro_visualization
pip install -e .
cd ../..
```

## 설치 검증

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')

import pytorch3d
print(f'PyTorch3D: OK')

import pointnet2_ops
print(f'pointnet2_ops: OK')

import curobo
print(f'CuRobo: OK')

import kaolin
print(f'Kaolin: OK')

# GPU 연산 테스트
x = torch.randn(100, 100, device='cuda')
y = x @ x.T
print(f'CUDA 연산 테스트: OK ({y.shape})')
"
```

## 트러블슈팅

### `no kernel image is available for execution on the device`

sm_120 커널 없이 빌드된 CUDA 확장 모듈이 있습니다. 해당 패키지를 `TORCH_CUDA_ARCH_LIST="12.0"`으로 재빌드하세요.

```bash
# 어떤 패키지에서 발생하는지 확인 후 재빌드
export TORCH_CUDA_ARCH_LIST="12.0"
pip install -e <해당_패키지_경로> --no-build-isolation --force-reinstall --no-deps
```

### PyTorch3D 빌드 실패

```bash
# 메모리 부족 시 병렬 빌드 제한
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Kaolin 빌드 시 torch 버전 에러

`IGNORE_TORCH_VER=1` 환경변수가 설정되어 있는지 확인하세요.

### CUDA 버전 불일치 경고

시스템 CUDA Toolkit(예: 13.0)과 PyTorch 번들 CUDA(12.8)가 다를 수 있습니다. 드라이버가 상위 호환되므로 정상 동작합니다. CUDA 확장 빌드 시 시스템의 nvcc가 사용되며, PyTorch의 CUDA 런타임과 링크됩니다.
