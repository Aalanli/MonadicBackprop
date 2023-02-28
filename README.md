# MonadicBackprop
## Engineering Choices
see main.rs

## Autograd as a Monad with linearity rules
I especially like jax's and flux-ml's approach to backprop, the former is functional, where there is some functor transforming $f$ to $f^\prime$, while the latter register's the backwards pass as a closure, which is what this technique relies on.

Consider $f: a \to b$, a pure function. Then the backwards operation would be $f^\prime: \nabla b \to \nabla a$, from the tangent space of $b$ to the tangent space of $a$. For convenience sake we write $f^\prime: b \to a$ to mean the same thing, since the derivative maps vectors from the output space to the input space.

Then in flux-ml, a custom differentiable operation can be registered as $f: a \to (b, f^\prime: b \to a)$, or in haskell notation $f: a \to (b, b \to a)$. This seems quite familar, we can write $(b, b \to a)$ as $m \; b$.

So a forward operation is $$f: a \to m\: b$$
Then say there is another differentiable function $g: b \to m\; c$. then what is the derivative of the composed function? 
Expanding the definitions, we get $f: a \to (b, \nabla b \to \nabla a)$, $g: b \to (c, \nabla c \to \nabla b)$. Then we want to evaluate $f$ first on $a$, and $g$ on the output of $f$, with the backwards function being $\nabla c \to \nabla a$.

So we want to pass $(b, \nabla b \to \nabla a)$, or $m\; b$ to $g: b \to (c, \nabla c \to \nabla b)$, or $g: b \to m\; c$, and get $(c, \nabla c \to \nabla a)$. Where have I seen that before? The bind operator for a Monad! $$F: m\; a \to (a \to m\;b) \to m \;b $$
In this case, the monad is a pair, where the first element is the image, and the second is the derivative.

```haskell
(>>=) (a, f') g = let (b, g') = g a
				  in (b, f' . g')
```

And bind is simply the chain rule.

However, some difficulties arises from haskell's standard monad. Consider $f: a \to m\; b$, $g: a\to m \; c$ and $h: (b, c) \to m \; d$




## Rambles
Upon trying multiple back-propagation libraries for the rust programming language, I have come across a common limitation. That is, trying to replicate the syntax and semantics of pytorch, which does not work quite as well in rust. 

Syntaxically, the notational burden of pytorch is quite light; there is no need to explicitly define the backwards pass, one simply has to use regular mathematical expressions to define the forwards pass. To make this happen in rust, however, exposes some tough design choices.
Since one would need to store some intermediate activations, and ideally, one would want 'normal' forwards notation ($c = a + b$), one would need to store the gradient tape in the tensors themselves. However, ownership complicates things, as intermediate activations go out of scope, so the hidden 'gradient tape' would need to take ownership, but the tensors themselves would need to have value semantics, to avoid specifying annoying lifetimes. Usually, this would involve some sort of Arc pointer, with every copy, or new tensor created sharing this pointer. A crate dfdx takes a difficult and different approach by constraining operations to only produce one output. In practice this may not be an issue for standard implimentations, but where both approaches in rust falls short is specifying custom operations. The former case makes this difficult as one would need to dissect the gradient tape to register a new backwards op. Additionally, a common theme, it appears to me, is that the backwards pass is buried under many layers of traits and generic definitions, further increasing the difficulty of extension.

In addition, all the libraries I looked at seem to have a variant of pytorch's gradient tape semantics, under which there is a clearly delinated forwards and backwards pass, with one syncronously following the other, and some sort of gradient accessing pattern.