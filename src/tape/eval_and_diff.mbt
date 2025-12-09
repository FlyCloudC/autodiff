///|
pub fn[A] Loc::at(self : Loc[A], mem : Array[A]) -> A {
  match self {
    Const(x) => x
    Memory(i) => mem[i]
  }
}

///|
pub fn[A] Tape::eval(self : Tape[A]) -> Array[A] {
  let insts = self.insts
  // m is memory
  // m[i] = x_i
  let m = Array::new(capacity=insts.length())
  for inst in insts {
    m.push(
      match inst {
        Var(x) => x
        App1(op1, arg_loc, ..) => op1(arg_loc.at(m))
        App2(op2, arg_l_loc, arg_r_loc, ..) =>
          op2(arg_l_loc.at(m), arg_r_loc.at(m))
      },
    )
  }
  m
}

///|
pub fn[A : Diffable] Tape::diff_forward(
  self : Tape[A],
  m : Array[A],
  wrt? : Int = 0,
) -> Array[A] {
  let insts = self.insts
  // d is diff memory.
  // d[i] = d(x_i) / d(x_wrt)
  let d = Array::make(insts.length(), A::zero())
  d[wrt] = A::one()
  for cur_i in (wrt + 1)..<insts.length() {
    let cur_inst = insts[cur_i]
    match cur_inst {
      App1(_, Memory(arg_i), diff~) => {
        let arg = m[arg_i]
        d[cur_i] = diff(arg) * d[arg_i]
      }
      App2(_, Memory(arg_l_i), Memory(arg_r_i), diff_l~, diff_r~) => {
        let arg_l = m[arg_l_i]
        let arg_r = m[arg_r_i]
        d[cur_i] = diff_l(arg_l, arg_r) * d[arg_l_i] +
          diff_r(arg_l, arg_r) * d[arg_r_i]
      }
      App2(_, Memory(arg_l_i), Const(arg_r), diff_l~, ..) =>
        d[cur_i] = diff_l(m[arg_l_i], arg_r) * d[arg_l_i]
      App2(_, Const(arg_l), Memory(arg_r_i), diff_r~, ..) =>
        d[cur_i] = diff_r(arg_l, m[arg_r_i]) * d[arg_r_i]
      Var(_) | App1(_, Const(_), ..) | App2(_, Const(_), Const(_), ..) => ()
    }
  }
  d
}

///|
pub fn[A : Diffable] Tape::diff_backward(
  self : Tape[A],
  m : Array[A],
  wrt? : Int = -1,
) -> Array[A] {
  let insts = self.insts
  let wrt = if wrt < 0 { insts.length() + wrt } else { wrt }
  // d is diff memory.
  // d[i] = d(x_wrt) / d(x_i)
  let d = Array::make(insts.length(), A::zero())
  d[wrt] = A::one()
  for cur_i = wrt; cur_i >= 0; cur_i = cur_i - 1 {
    let cur_inst = insts[cur_i]
    match cur_inst {
      App1(_, Memory(arg_i), diff~) => d[arg_i] += diff(m[arg_i]) * d[cur_i]
      App2(_, Memory(arg_i_1), Memory(arg_i_2), diff_l~, diff_r~) => {
        let arg_1 = m[arg_i_1]
        let arg_2 = m[arg_i_2]
        let d_cur = d[cur_i]
        d[arg_i_1] += diff_l(arg_1, arg_2) * d_cur
        d[arg_i_2] += diff_r(arg_1, arg_2) * d_cur
      }
      App2(_, Memory(arg_i_1), Const(arg_2), diff_l~, ..) =>
        d[arg_i_1] += diff_l(m[arg_i_1], arg_2) * d[cur_i]
      App2(_, Const(arg_1), Memory(arg_i_2), diff_r~, ..) =>
        d[arg_i_2] += diff_r(arg_1, m[arg_i_2]) * d[cur_i]
      Var(_) | App1(_, Const(_), ..) | App2(_, Const(_), Const(_), ..) => ()
    }
  }
  d
}
