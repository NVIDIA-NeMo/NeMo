#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while read testcase; do
    IFS='~' read spoken written <<< $testcase
    denorm_pred=$(echo $spoken | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    # trim white space
    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}

testITNCardinal() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testITNDate() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_date.txt
  runtest $input
}

testITNDecimal() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $input
}

testITNOrdinal() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $input
}

#testITNFraction() {
#  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_fraction.txt
#  runtest $input
#}

testITNTime() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}

testITNMeasure() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_measure.txt
  runtest $input
}

testITNMoney() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_money.txt
  runtest $input
}

testITNWhitelist() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $input
}

testITNTelephone() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_telephone.txt
  runtest $input
}

testITNElectronic() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_electronic.txt
  runtest $input
}

testITNWord() {
  input=$PROJECT_DIR/pt/data_inverse_text_normalization/test_cases_word.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
