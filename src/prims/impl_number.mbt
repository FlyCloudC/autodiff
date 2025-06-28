// -----------------
//       Float
// -----------------

///|
pub impl Number for Float with exp(self) {
  @math.expf(self)
}

///|
pub impl Number for Float with ln(self) {
  @math.lnf(self)
}

///|
pub impl Number for Float with sin(self) {
  @math.sinf(self)
}

///|
pub impl Number for Float with cos(self) {
  @math.cosf(self)
}

// -----------------
//       Double
// -----------------

///|
pub impl Number for Double with exp(self) {
  @math.exp(self)
}

///|
pub impl Number for Double with ln(self) {
  @math.ln(self)
}

///|
pub impl Number for Double with sin(self) {
  @math.sin(self)
}

///|
pub impl Number for Double with cos(self) {
  @math.cos(self)
}
