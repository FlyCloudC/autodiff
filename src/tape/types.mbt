///|
pub enum Loc[A] {
  Const(A)
  Memory(Int)
} derive(Show)

///|
type Op1[A] = (A) -> A

///|
type Op2[A] = (A, A) -> A

///|
pub(all) enum Inst[A] {
  Var(A)
  App1(Op1[A], Loc[A], diff~ : Op1[A])
  App2(Op2[A], Loc[A], Loc[A], diff_l~ : Op2[A], diff_r~ : Op2[A])
}

///|
pub(all) struct Tape[A] {
  insts : Array[Inst[A]]
  names : Array[String]
}

///|
pub(open) trait Diffable: Add + Mul {
  zero() -> Self
  one() -> Self
}
