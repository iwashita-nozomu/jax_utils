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
    @property
    def shape(self) -> Tuple[int,...]:
        if self.__shape__ is None:
            raise ValueError("Shape is not specified.")
        return self.__shape__
    
    def __init__(self,mv:Callable[[Vector],Vector],shape:Optional[Tuple[int,...]]=None):
        self.mv = self._ensure_batched(mv)
        self.__shape__ = shape

    def _ensure_batched(self,mv:Callable[[Vector],Vector])->Callable[[Matrix],Matrix]:
        def batched_mv(x:Array)->Array:
            if x.ndim == 1:
                return mv(x)
            elif x.ndim ==2:
                return eqx.filter_vmap(mv,in_axes=1,out_axes=1)(x)
            else:
                raise ValueError("Input array must be 1D or 2D.")
        return batched_mv
    
    @overload
    def __matmul__(self,x:Matrix,/)->Matrix: ...
    @overload
    def __matmul__(self,x:LinearOperator,/)->LinearOperator: ...

    def __matmul__(self,x:Array|LinearOperator,/)->Array|LinearOperator:
        if isinstance(x,jax.Array):
            return self.mv(x)
        else:
            def composed_mv(v:Array)->Array:
                return self @ (x @ v)
            return LinOp(composed_mv)
        
    @overload
    def __mul__(self,other:Array,/)->LinearOperator: ...
    @overload
    def __mul__(self,other:LinearOperator,/)->LinearOperator: ...

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
            def composed_mv(v:Array)->Array:
                return self @ (other @ v)
            return LinOp(composed_mv)
    @overload
    def __rmul__(self,other:Array,/)->LinearOperator: ...
    @overload
    def __rmul__(self,other:LinearOperator,/)->LinearOperator: ...

    def __rmul__(self,other:Array|LinearOperator) -> LinearOperator:
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
    
    def __add__(self,other:LinearOperator,/)->LinearOperator:
        return LinOp(lambda v: self @ v + other @ v,shape=self.shape)


def hstack_linops(ops:List[LinearOperator])->LinearOperator:

    v_dims = [op.shape[1] for op in ops]
    u_dims = [op.shape[0] for op in ops]
    for op in ops:
        if op.shape[0] != u_dims[0]:
            raise ValueError("All LinearOperators must have the same input dimension for hstack.")

    def hstack_mv(v:Vector)->Vector:
        results = []
        cnt = 0
        for op,dim in zip(ops, v_dims):
            results.append(op @ v[cnt:cnt+dim])
            cnt += dim
        return jnp.sum(jnp.stack(results),axis=0)
    return LinOp(hstack_mv,shape=(u_dims[0],sum(v_dims)))

def vstack_linops(ops:List[LinearOperator])->LinearOperator:

    for op in ops:
        if op.shape[1] != ops[0].shape[1]:
            raise ValueError("All LinearOperators must have the same output dimension for vstack.")
    def vstack_mv(v:Vector)->Vector:
        results = []
        for op in ops:
            results.append(op @ v)
        return jnp.concatenate(results,axis=0)
    total_u_dim = sum([op.shape[0] for op in ops])
    return LinOp(vstack_mv,shape=(total_u_dim,ops[0].shape[1]))

def stack_linops(ops:List[List[LinearOperator]])->LinearOperator:
    vops = [hstack_linops(row) for row in ops]
    return vstack_linops(vops)

__all__ = [
    "LinOp",
]