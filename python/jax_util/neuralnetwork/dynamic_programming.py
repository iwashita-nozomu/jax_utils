'''
各レイヤーで、価値関数V_kを構築。
解レイヤーの更新後、再帰的に上位も更新する必要がある。
'''

from __future__ import annotations

from ..base import OptimizationProblem,ConstrainedOptimizationProblem

from ..functional import Function

from .protocols import *

from .neuralnetwork import *

import equinox as eqx

from typing import Tuple, Any

import jax

class SingleLayerDynamicProgramming(eqx.Module):
    # 再帰を実行するためのクラス
    upper_layer_optimizer: "SingleLayerDynamicProgramming"
    layer_static: Static
    
    def __call__(
            self,
            layer_param: Params, # f_k
            lower_model: Function, # phi_k-1
    ) -> Tuple[Params, PyTreeOptimizationProblem,Aux]:
        # そのレイヤーのパラメータに関する最適化問題に変換する
        

def initiakize_dp_trainer(
        model: NeuralNetwork,
) -> Tuple[SingleLayerDynamicProgramming,...]:
    
    for i in range

def train(
        model: NeuralNetwork,
        optim: OptimizationProblem[NeuralNetwork],
        context: Tuple[Any,...],#積分器、学習率スケジューラーなどの追加情報
) -> Tuple[NeuralNetwork, Aux]:
    
    if isinstance(optim, ConstrainedOptimizationProblem):
        raise NotImplementedError("制約付き最適化問題は未実装です。")
    
    # まずは、再帰を行うクラスを構築する。

    

    # その後、最下層を呼び出せば完了！