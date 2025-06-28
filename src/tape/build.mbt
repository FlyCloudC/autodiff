///|
pub fn[A] Tape::new() -> Tape[A] {
  { insts: [], names: [] }
}

///|
pub fn[A] constant(x : A) -> Loc[A] {
  Const(x)
}

///|
pub fn[A] variable(self : Tape[A], x : A) -> Loc[A] {
  let { insts, names } = self
  insts.push(Var(x))
  names.push("var")
  Memory(insts.length() - 1)
}

///|
pub fn[A] op1(
  self : Tape[A],
  name : String,
  op : Op1[A],
  diff : Op1[A]
) -> Op1[Loc[A]] {
  x => {
    let { insts, names } = self
    insts.push(App1(op, x, diff~))
    names.push(name)
    Memory(insts.length() - 1)
  }
}

///|
pub fn[A] op2(
  self : Tape[A],
  name : String,
  op : Op2[A],
  diff_l : Op2[A],
  diff_r : Op2[A]
) -> Op2[Loc[A]] {
  (l, r) => {
    let { insts, names } = self
    insts.push(App2(op, l, r, diff_l~, diff_r~))
    names.push(name)
    Memory(insts.length() - 1)
  }
}
