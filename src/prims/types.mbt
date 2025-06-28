///|
typealias (A) -> A as Op1[A]

///|
typealias (A, A) -> A as Op2[A]

///|
typealias @tape.(Tape, Loc)

///|
pub(open) trait Number: @tape.Diffable + Add + Mul + Neg + Sub + Div {
  exp(Self) -> Self
  ln(Self) -> Self
  sin(Self) -> Self
  cos(Self) -> Self
}
