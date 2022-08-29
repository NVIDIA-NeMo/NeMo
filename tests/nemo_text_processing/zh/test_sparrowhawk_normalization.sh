#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while read testcase; do
    IFS='~' read written spoken <<< $testcase
    # replace non breaking space with breaking space
    denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    # # trim white space
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNMoneyText() {
  input=$PROJECT_DIR/zh/data_text_normalization/test_cases_money.txt
  runtest $input
}
testTNCharText() {
  input=$PROJECT_DIR/zh/data_text_normalization/test_cases_char.txt
  runtest $input
}
testTNTimeText() {
  input=$PROJECT_DIR/zh/data_text_normalization/test_cases_time.txt
  runtest $input
}
testTNDateText() {
  input=$PROJECT_DIR/zh/data_text_normalization/test_cases_date.txt
  runtest $input
}
# testTNMathText() {
#   input=$PROJECT_DIR/zh/data_text_normalization/test_cases_math.txt
#   runtest $input
# }
testTNFractionText() {
  input=$PROJECT_DIR/zh/data_text_normalization/test_cases_fraction.txt
  runtest $input
}

# testTNPreprocessText() {
#   input=$PROJECT_DIR/zh/data_text_normalization/test_cases_preprocess.txt
#   runtest $input
# }
testTNMeasureText() {
  input=$PROJECT_DIR/zh/data_text_normalization/test_cases_measure.txt
  runtest $input
}


# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
