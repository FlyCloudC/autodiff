///|
pub fn[A : Show] Tape::dump(
  self : Tape[A],
  pad_1~ : Int = 4,
  pad_2~ : Int = 4
) -> String {
  let loc_name = (loc : Loc[A]) => match loc {
    Const(x) => x.to_string()
    Memory(i) => "x\{i}"
  }
  let buf = StringBuilder::new()
  for i, inst in self.insts {
    buf.write_string("x\{i}=".pad_end(4, ' '))
    match inst {
      Var(x) => buf.write_object(x)
      App1(_, loc, ..) =>
        buf
        ..write_string(self.names[i].pad_end(pad_1, ' '))
        ..write_string(loc_name(loc))
      App2(_, loc1, loc2, ..) =>
        buf
        ..write_string(self.names[i].pad_end(pad_1, ' '))
        ..write_string(loc_name(loc1).pad_end(pad_2, ' '))
        ..write_string(loc_name(loc2))
    }
    buf.write_char('\n')
  }
  buf.to_string()
}
