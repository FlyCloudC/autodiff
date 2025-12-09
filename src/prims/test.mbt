///|
test "tape" {
  let tape : Tape[Double] = Tape::new()
  let { neg, add, mul, .. } = AllPrims::on(tape)
  let x0 = tape.variable(2.0)
  let x1 = tape.variable(5.0)
  let res = add(neg(x0), mul(x0, x1))

  // Now the instructions are recorded on the tape
  inspect(
    tape.dump(pad_1=2, pad_2=3),
    content=(
      #|x0= 2
      #|x1= 5
      #|x2= - x0
      #|x3= * x0 x1
      #|x4= + x2 x3
      #|
    ),
  )

  // Eval
  let mem = tape.eval()
  inspect(mem, content="[2, 5, -2, 10, 8]")
  guard res is Memory(res_index)
  inspect(mem[res_index], content="8")

  // Forward differentiation with respect to x0
  let diff_forward_mem_0 = tape.diff_forward(mem, wrt=0)
  inspect(diff_forward_mem_0, content="[1, 0, -1, 5, 4]")

  // Forward differentiation with respect to x1
  let diff_forward_mem_1 = tape.diff_forward(mem, wrt=1)
  inspect(diff_forward_mem_1, content="[0, 1, 0, 2, 2]")

  // Backwood differentiation with respect to x1
  let diff_backward_mem = tape.diff_backward(mem)
  inspect(diff_backward_mem, content="[4, 2, 1, 1, 1]")
}
