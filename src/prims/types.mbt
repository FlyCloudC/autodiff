///|
type Op1[A] = (A) -> A

///|
type Op2[A] = (A, A) -> A

///|
using @tape {type Tape, type Loc}

///|
pub(open) trait Number: @tape.Diffable + Add + Mul + Neg + Sub + Div {
  exp(Self) -> Self
  ln(Self) -> Self
  sin(Self) -> Self
  cos(Self) -> Self
}
