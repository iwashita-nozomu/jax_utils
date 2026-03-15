from __future__ import annotations


from .protocols import *

import equinox as eqx

from typing import Callable,overload,List,Tuple,Optional
import jax
from jaxtyping import Array

from jax import numpy as jnp

class LinOp(eqx.Module):
    mv : Callable[[Matrix],Matrix]
    __shape__: Optional[Tuple[int,...]] = eqx.field(static=True)

    # 責務: 既知なら作用素の shape を公開します。
    @property
    def shape(self) -> Tuple[int,...]:
        if self.__shape__ is None:
            raise ValueError("Shape is not specified.")
        return self.__shape__
    
    # 責務: 単一ベクトル向けの matvec を batched 対応の線形作用素へ包みます。
    def __init__(self,mv:Callable[[Vector],Vector],shape:Optional[Tuple[int,...]]=None):
        self.mv = self._ensure_batched(mv)
        self.__shape__ = shape

    # 責務: `Vector -> Vector` の関数を `Matrix` 入力も受ける形へ拡張します。
    def _ensure_batched(self,mv:Callable[[Vector],Vector])->Callable[[Matrix],Matrix]:
        # 責務: 1D/2D 入力を吸収して共通の matvec として実行します。
        def batched_mv(x:Array)->Array:
            if x.ndim == 1:
                return mv(x)
            elif x.ndim ==2:
                return eqx.filter_vmap(mv,in_axes=1,out_axes=1)(x)
            else:
                raise ValueError("Input array must be 1D or 2D.")
        return batched_mv
    
    # 責務: `@` 演算の型ごとの返り値契約を宣言する。
    @overload
    def __matmul__(self,x:Matrix,/)->Matrix: ...
    # 責務: 線形作用素どうしを合成したときの返り値契約を宣言する。
    @overload
    def __matmul__(self,x:LinearOperator,/)->LinearOperator: ...

    # 責務: ベクトル適用または線形作用素の合成を `@` で表現します。
    def __matmul__(self,x:Array|LinearOperator,/)->Array|LinearOperator:
        if isinstance(x,jax.Array):
            return self.mv(x)
        else:
            # 責務: 右側の線形作用素を先に適用する合成 matvec を作ります。
            def composed_mv(v:Array)->Array:
                return self @ (x @ v)
            return LinOp(composed_mv)
        
    # 責務: `*` 演算の型ごとの返り値契約を宣言する。
    @overload
    def __mul__(self,other:Array,/)->LinearOperator: ...
    # 責務: 作用素合成としての `*` 演算の返り値契約を宣言する。
    @overload
    def __mul__(self,other:LinearOperator,/)->LinearOperator: ...

    # 責務: スカラー倍または線形作用素の合成を `*` で表現します。
    def __mul__(self,other:Array|LinearOperator) -> LinearOperator:
        if isinstance(other,jax.Array):

            if other.ndim == 0:
                return LinOp(lambda v: other * (self @ v))
            
            elif other.ndim ==1:
                raise ValueError(f"Scalar or LinearOperator expected for multiplication.{other.__name__} is vector.")
            
            elif other.ndim ==2:
                return LinOp(lambda v: (self @ other) @ v)
            
            else:
                raise ValueError(f"Scalar or LinearOperator expected for multiplication.{other.__name__} is {other.ndim}D array.")

        else:
            # 責務: 右側の線形作用素を先に適用する合成 matvec を作ります。
            def composed_mv(v:Array)->Array:
                return self @ (other @ v)
            return LinOp(composed_mv)
    # 責務: 左側からの `*` 演算の型ごとの返り値契約を宣言する。
    @overload
    def __rmul__(self,other:Array,/)->LinearOperator: ...
    # 責務: 左作用素との合成としての `*` 演算の返り値契約を宣言する。
    @overload
    def __rmul__(self,other:LinearOperator,/)->LinearOperator: ...

    # 責務: 左スカラー倍または左作用素との合成を `*` で表現します。
    def __rmul__(self,other:Array|LinearOperator) -> LinearOperator:
        # 責務: 左側の作用を先に反映した合成 matvec を作ります。
        def composed_mv(v:Array)->Array:
            if isinstance(other,jax.Array):
                if other.ndim == 0:
                    return other * (self @ v)
                elif other.ndim ==1:
                    raise ValueError(f"Scalar or LinearOperator expected for multiplication.{other.__name__} is vector.")
                elif other.ndim ==2:
                    return other @ (self @ v)
                else:
                    raise ValueError(f"Scalar or LinearOperator expected for multiplication.{other.__name__} is {other.ndim}D array.")
                
            return other @ (self @ v)
        
        return LinOp(composed_mv)
    
    # 責務: 2 つの線形作用素の和を新しい線形作用素として返します。
    def __add__(self,other:LinearOperator,/)->LinearOperator:
        return LinOp(lambda v: self @ v + other @ v,shape=self.shape)


# 責務: 共通の出力次元を持つ線形作用素を横方向に連結します。
def hstack_linops(ops:List[LinearOperator])->LinearOperator:

    v_dims = [op.shape[1] for op in ops]
    u_dims = [op.shape[0] for op in ops]
    for op in ops:
        if op.shape[0] != u_dims[0]:
            raise ValueError("All LinearOperators must have the same input dimension for hstack.")

    # 責務: 分割した入力ベクトルへ各作用素を適用し、結果を合算します。
    def hstack_mv(v:Vector)->Vector:
        results = []
        cnt = 0
        for op,dim in zip(ops, v_dims):
            results.append(op @ v[cnt:cnt+dim])
            cnt += dim
        return jnp.sum(jnp.stack(results),axis=0)
    return LinOp(hstack_mv,shape=(u_dims[0],sum(v_dims)))

# 責務: 共通の入力次元を持つ線形作用素を縦方向に連結します。
def vstack_linops(ops:List[LinearOperator])->LinearOperator:

    for op in ops:
        if op.shape[1] != ops[0].shape[1]:
            raise ValueError("All LinearOperators must have the same output dimension for vstack.")
    # 責務: 同じ入力へ各作用素を適用し、出力を縦に積みます。
    def vstack_mv(v:Vector)->Vector:
        results = []
        for op in ops:
            results.append(op @ v)
        return jnp.concatenate(results,axis=0)
    total_u_dim = sum([op.shape[0] for op in ops])
    return LinOp(vstack_mv,shape=(total_u_dim,ops[0].shape[1]))

# 責務: 2 次元ブロックの線形作用素配列を 1 つの作用素へまとめます。
def stack_linops(ops:List[List[LinearOperator]])->LinearOperator:
    vops = [hstack_linops(row) for row in ops]
    return vstack_linops(vops)

__all__ = [
    "LinOp",
]
