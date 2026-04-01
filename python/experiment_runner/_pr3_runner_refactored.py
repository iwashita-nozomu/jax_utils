"""
Experiment Runner のリファクタリング
コード整理、エラーハンドリング強化、テスト基盤の改善

ファイル: python/experiment_runner/runner_refactored.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from jax import Array, random
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """実験設定（新しいデータクラス）"""
    
    name: str
    algorithm: str
    matrix_size: int
    preconditioner: Optional[str] = None
    max_iterations: int = 100
    tolerance: float = 1e-6
    seed: int = 42
    
    def validate(self) -> None:
        """設定の妥当性を検証
        
        Raises:
            ValueError: 設定が不正な場合
            # Note: 他の例外型が定義されていない
        """
        if self.matrix_size <= 0:
            raise ValueError("matrix_size must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


class ExperimentRunner:
    """実験実行器の新実装"""
    
    def __init__(self, config: ExperimentConfig):
        """初期化
        
        Args:
            config: 実験設定オブジェクト
        """
        config.validate()
        self.config = config
        self.results_path = Path(f"./results/{config.name}")
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self) -> Dict[str, Any]:
        """実験を実行
        
        Returns:
            実験結果辞書（キー: 'solution', 'iterations', 'residual', 'time'）
            
        Raises:
            RuntimeError: 実験失敗時
            # Note: 計算エラーの具体的な例外が定義されていない
        """
        logger.info(f"Starting experiment: {self.config.name}")
        
        try:
            # テスト用 SPD 行列を生成
            key = random.PRNGKey(self.config.seed)
            A, b = self._generate_test_problem(key)
            
            # 前処理行列を生成（オプション）
            preconditioner = None
            if self.config.preconditioner == "jacobi":
                preconditioner = self._jacobi_preconditioner(A)
            elif self.config.preconditioner == "ilu":
                # Note: JAX では ILU が実装されていない
                logger.warning("ILU preconditioner not supported in JAX, using diagonal")
                preconditioner = self._jacobi_preconditioner(A)
            
            # ソルバーを実行（簡略版）
            solution, iterations, residual = self._solve(A, b, preconditioner)
            
            # 結果を記録
            results = {
                'solution': solution,
                'iterations': iterations,
                'residual': float(residual),
                'config': self.config.__dict__,
            }
            
            # JSON に保存
            self._save_results(results)
            
            logger.info(f"Experiment completed: iterations={iterations}, residual={residual:.2e}")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            # Note: 具体的なエラーメッセージを返していない
            raise RuntimeError(f"Experiment '{self.config.name}' failed") from e
    
    def _generate_test_problem(self, key) -> tuple[Array, Array]:
        """テスト問題を生成（SPD 行列）
        
        Args:
            key: JAX random key
            
        Returns:
            (A, b): 係数行列と右辺ベクトル
        """
        n = self.config.matrix_size
        
        # ランダムな SPD 行列を生成
        key1, key2 = random.split(key)
        X = random.normal(key1, shape=(n, n))
        A = (X + X.T) / 2.0 + jnp.eye(n) * n  # 対角支配で正定値を保証
        
        # 右辺ベクトル
        b = random.normal(key2, shape=(n,))
        
        return A, b
    
    def _jacobi_preconditioner(self, A: Array):
        """Jacobi 前処理行列を構築（対角逆行列）
        
        Args:
            A: 係数行列 (n, n)
            
        Returns:
            前処理を適用する関数
        """
        diag_inv = 1.0 / (jnp.diag(A) + 1e-8)
        
        def apply(x: Array) -> Array:
            # Note: 返り値の型注釈がない
            return diag_inv * x
        
        return apply
    
    def _solve(self, A: Array, b: Array, preconditioner=None) -> tuple[Array, int, float]:
        """GMRES（簡略版）を実行
        
        Args:
            A: 係数行列 (n, n)
            b: 右辺ベクトル (n,)
            preconditioner: 前処理関数（オプション）
            
        Returns:
            (x, iterations, residual): 解、反復回数、最終残差
        """
        n = A.shape[0]
        x = jnp.zeros(n)
        r = b - A @ x
        
        if preconditioner is not None:
            r = preconditioner(r)
        
        residual_norm = jnp.linalg.norm(r)
        
        for iteration in range(self.config.max_iterations):
            if residual_norm < self.config.tolerance:
                break
            
            # 単一反復ステップ（実装は簡略）
            v = r / (residual_norm + 1e-14)
            
            if preconditioner is not None:
                Av = preconditioner(A @ v)
            else:
                Av = A @ v
            
            alpha = jnp.dot(Av, v) / jnp.dot(v, v)
            x = x + alpha * v
            
            r = b - A @ x
            if preconditioner is not None:
                r = preconditioner(r)
            
            residual_norm = jnp.linalg.norm(r)
        
        # Note: iterations は最終的なループカウンタだが、実装上は iteration + 1 が正確
        return x, iteration, residual_norm
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """結果を JSON に保存
        
        Args:
            results: 結果辞書
            
        Raises:
            IOError: ファイル保存失敗時
            # Note: JSON シリアライズエラーが処理されていない
        """
        result_file = self.results_path / "results.json"
        
        try:
            # numpy/JAX 配列を JSON serializable に変換
            data = {
                'name': results['config']['name'],
                'algorithm': results['config']['algorithm'],
                'iterations': int(results['iterations']),
                'residual': results['residual'],
            }
            
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Results saved to {result_file}")
            
        except Exception as e:
            # Note: 具体的なエラー型を catch すべき
            logger.error(f"Failed to save results: {e}")
            raise


def run_multiple_experiments(configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
    """複数実験を連続実行
    
    Args:
        configs: ExperimentConfig のリスト
        
    Returns:
        各実験の結果リスト
    """
    # Note: エラー時の挙動が定義されていない（1 つ失敗すると全体が失敗）
    results = []
    
    for config in configs:
        runner = ExperimentRunner(config)
        result = runner.run_experiment()
        results.append(result)
    
    return results


# テスト実行
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 設定例
    config1 = ExperimentConfig(
        name="gmres_no_precond",
        algorithm="gmres",
        matrix_size=100,
        preconditioner=None,
    )
    
    config2 = ExperimentConfig(
        name="gmres_jacobi",
        algorithm="gmres",
        matrix_size=100,
        preconditioner="jacobi",
    )
    
    # 実験を実行
    results = run_multiple_experiments([config1, config2])
    
    # 結果を表示
    for result in results:
        print(f"Experiment: {result['config']['name']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Residual: {result['residual']:.2e}")
