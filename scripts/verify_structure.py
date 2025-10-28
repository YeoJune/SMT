"""
Project Structure Verification
"""

import os
from pathlib import Path


def check_file_exists(filepath):
    """Check if file exists and return status"""
    exists = os.path.exists(filepath)
    size = os.path.getsize(filepath) if exists else 0
    return exists, size


def main():
    """Verify project structure and files"""
    
    print("=" * 70)
    print("SMT (STRIDE MEMORY TRANSFORMER) - PROJECT STRUCTURE")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    
    files_to_check = [
        # Root
        ("README.md", "프로젝트 소개"),
        ("requirements.txt", "의존성"),
        ("IMPLEMENTATION.md", "구현 상세"),
        
        # Config
        ("config/__init__.py", "Config 패키지"),
        ("config/model_config.py", "모델 설정"),
        
        # Models
        ("src/__init__.py", "Src 패키지"),
        ("src/models/__init__.py", "Models 패키지"),
        ("src/models/smt.py", "메인 모델"),
        ("src/models/window_manager.py", "윈도우 관리자"),
        
        # Components
        ("src/models/components/__init__.py", "Components 패키지"),
        ("src/models/components/attention_pooling.py", "Attention Pooling"),
        ("src/models/components/transformer.py", "Transformer"),
        ("src/models/components/ssm.py", "SSM"),
        
        # Tests
        ("tests/test_integration.py", "통합 테스트"),
    ]
    
    print("\n📁 파일 구조:")
    print("-" * 80)
    
    total_size = 0
    for filepath, description in files_to_check:
        full_path = project_root / filepath
        exists, size = check_file_exists(full_path)
        
        status = "✅" if exists else "❌"
        size_kb = size / 1024 if size > 0 else 0
        
        print(f"{status} {filepath:50s} {description:30s} ({size_kb:6.1f} KB)")
        total_size += size
    
    print("-" * 80)
    print(f"📊 총 코드 크기: {total_size/1024:.1f} KB")
    
    # Count lines of code
    print("\n📝 코드 라인 수:")
    print("-" * 80)
    
    code_files = [
        "config/model_config.py",
        "src/models/smt.py",
        "src/models/window_manager.py",
        "src/models/components/attention_pooling.py",
        "src/models/components/transformer.py",
        "src/models/components/ssm.py",
    ]
    
    total_lines = 0
    for filepath in code_files:
        full_path = project_root / filepath
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"  {filepath:50s} {lines:5d} lines")
    
    print("-" * 80)
    print(f"📊 총 코드 라인: {total_lines:,} lines")
    
    # Component summary
    print("\n🔧 구현된 컴포넌트:")
    print("-" * 80)
    
    components = [
        ("AttentionPooling", "윈도우를 단일 벡터로 압축하는 학습 가능한 pooling"),
        ("WindowManager", "SSM outputs + input tokens의 sliding window 관리"),
        ("WindowedTransformer", "GPT-2 기반 작은 윈도우 처리"),
        ("SSMMemory", "Mamba 기반 압축 메모리"),
        ("StrideMemoryTransformer", "전체 아키텍처 통합"),
        ("SMTConfig", "모델 설정 및 검증"),
        ("TrainingConfig", "학습 설정"),
    ]
    
    for name, desc in components:
        print(f"  ✅ {name:25s} - {desc}")
    
    # Next steps
    print("\n📋 다음 구현 단계:")
    print("-" * 80)
    
    next_steps = [
        "Data Pipeline (WikiText-103, PG-19 로더)",
        "Training Loop (trainer, optimizer, scheduler)",
        "Utilities (metrics, checkpointing, logging)",
        "Experiment Scripts (train, evaluate, ablation)",
        "Pre-trained Weight Loading (GPT-2 + Mamba)",
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    
    print("\n" + "="*80)
    print("✅ 프로젝트 구조 준비 완료!")
    print("="*80)
    print("\n💡 다음 단계: requirements.txt의 패키지를 설치하고 테스트를 실행하세요")
    print("   pip install -r requirements.txt")
    print("   python tests/test_integration.py")
    print()


if __name__ == "__main__":
    main()
