# FlyCloudC/autodiff

The `autodiff` package provides automatic differentiation capabilities through a tape-based system. It allows recording mathematical operations and computing both forward and backward derivatives.

## Core Concepts

### Tape
The `Tape[A]` struct is the central component that records all operations. It contains:
- `insts : Array[Inst[A]]`: instructions
- `names : Array[String]`: instruction names for debugging

### Primitives
The `AllPrims[A]` struct provides common mathematical operations:
- Unary: `neg`, `exp`, `ln`, `sin`, `cos`
- Binary: `add`, `sub`, `mul`, `div`

### Location Types
The `Loc[A]` represents where a value is stored:
- `Const(A)`: Constant value
- `Memory(Int)`: Index into the memory array

## Key Functions

### Tape Initialization
```mbt
fn[A] Tape::new() -> Self[A]
```
Creates a new empty tape.

### Constant Creation
```mbt
fn[A] constant(x : A) -> Loc[A]
```
Creates a constant.

### Variable Creation
```mbt
fn[A] Tape::variable(Self[A], A) -> Loc[A]
```
Creates a new variable on the tape.

### Primitive Operations
```mbt
fn[A : Number] AllPrims::on(@tape.Tape[A]) -> Self[A]
```
Returns a set of primitive operations that can be recorded on the tape.

```mbt
fn[A] Tape::op1(Self[A], String, (A) -> A, (A) -> A) -> (Loc[A]) -> Loc[A]
fn[A] Tape::op2(Self[A], String, (A, A) -> A, (A, A) -> A, (A, A) -> A) -> (Loc[A], Loc[A]) -> Loc[A]
```
Register your own operations. For example, see [`prims/prims.mbt`](src/prims/prims.mbt)

### Evaluation
```mbt
fn[A] Tape::eval(Self[A]) -> Array[A]
```
Evaluates all recorded operations and returns the memory array.

The result of the i-th instruction is stored at the i-th position in the memory array.

### Differentiation
```mbt
fn[A : Diffable] Tape::diff_forward(Self[A], Array[A], wrt~ : Int = ..) -> Array[A]
fn[A : Diffable] Tape::diff_backward(Self[A], Array[A], wrt~ : Int = ..) -> Array[A]
```
Compute derivatives using forward or reverse mode automatic differentiation.

The `wrt` parameter (short for "with respect to") determines which variable's derivative is computed.

- In **forward mode** (`diff_forward`), the tape computes the derivative of every intermediate value with respect to the variable at index `wrt`. This is efficient when you need derivatives with respect to a single input variable.
- In **backward mode** (`diff_backward`), the tape computes the derivative of a single output (typically the last computed value) with respect to every input variable. This is efficient when you have many input variables but only one output.

### Diffable Trait

The `Diffable` trait is required for types that can be used with automatic differentiation:
```mbt
pub(open) trait Diffable : Add + Sub {
  zero() -> Self
  one() -> Self
}
```

Pre-implemented for `Float` and `Double`.

### Debugging
```mbt
fn[A : Show] Tape::dump(Self[A], pad_1~ : Int = .., pad_2~ : Int = ..) -> String
```
Returns a human-readable representation of the tape's contents.

## Example Usage

```mbt
let tape : Tape[Double] = Tape::new()
let { neg, add, mul, .. } = AllPrims::on(tape)
let x0 = tape.variable(2.0)
let x1 = tape.variable(5.0)
let res = add(neg(x0), mul(x0, x1))

// Evaluation
let mem = tape.eval()
inspect(mem, content="[2, 5, -2, 10, 8]")

// Dump
inspect(
  tape.dump(pad_1=2, pad_2=3),
  content=
    #|x0= 2
    #|x1= 5
    #|x2= - x0
    #|x3= * x0 x1
    #|x4= + x2 x3
    #|
  ,
)

// Forward differentiation with respect to x1
let diff_forward_mem_1 = tape.diff_forward(mem, wrt=1)
inspect(diff_forward_mem_1, content="[0, 1, 0, 2, 2]")

// Backward differentiation
let diff_backward_mem = tape.diff_backward(mem)
inspect(diff_backward_mem, content="[4, 2, 1, 1, 1]")
```

## Future Work

Extend this package by implementing a `Tensor[A]` type to support multi-dimensional arrays. To enable automatic differentiation for tensors, implement the `Diffable` trait for `Tensor[A]`, and register tensor operations using `Tape::op1` and `Tape::op2`. This will allow you to use all tape-based differentiation features with tensor operations.

This approach enables differentiable programming for vectorized and matrix computations.

## License

MIT
