///|
pub(all) struct AllPrims[A] {
  neg : Op1[Loc[A]]
  exp : Op1[Loc[A]]
  sin : Op1[Loc[A]]
  cos : Op1[Loc[A]]
  ln : Op1[Loc[A]]
  add : Op2[Loc[A]]
  sub : Op2[Loc[A]]
  mul : Op2[Loc[A]]
  div : Op2[Loc[A]]
}

///|
pub fn[A : Number] AllPrims::on(tape : Tape[A]) -> AllPrims[A] {
  let one = A::one()
  {
    neg: tape.op1("-", x => -x, _ => -one),
    exp: tape.op1("exp", A::exp, A::exp),
    ln: tape.op1("ln", A::ln, x => one / x),
    sin: tape.op1("sin", A::sin, A::cos),
    cos: tape.op1("cos", A::cos, x => -A::sin(x)),
    add: tape.op2("+", A::op_add, (_, _) => one, (_, _) => one),
    sub: tape.op2("=", A::op_sub, (_, _) => one, (_, _) => -one),
    mul: tape.op2("*", A::op_mul, (_, r) => r, (l, _) => l),
    div: tape.op2("/", A::op_div, (_, r) => one / r, (l, r) => -l / (r * r)),
  }
}
