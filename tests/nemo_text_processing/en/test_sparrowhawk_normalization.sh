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

    # trim white space
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNSpecialText() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_special_text.txt
  runtest $input
}

testTNCardinal() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testTNDate() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_date.txt
  runtest $input
}

testTNDecimal() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_decimal.txt
  runtest $input
}

testTNRange() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_range.txt
  runtest $input
}

testTNSerial() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_serial.txt
  runtest $input
}

#testTNRoman() {
#  input=$PROJECT_DIR/en/data_text_normalization/test_cases_roman.txt
#  runtest $input
#}

testTNElectronic() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_electronic.txt
  runtest $input
}

testTNFraction() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_fraction.txt
  runtest $input
}

testTNMoney() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_money.txt
  runtest $input
}

testTNOrdinal() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_ordinal.txt
  runtest $input
}

testTNTelephone() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_telephone.txt
  runtest $input
}

testTNTime() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_time.txt
  runtest $input
}

testTNMeasure() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_measure.txt
  runtest $input
}

testTNWhitelist() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_whitelist.txt
  runtest $input
}

testTNWord() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_word.txt
  runtest $input
}

testTNAddress() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_address.txt
  runtest $input
}

testTNMath() {
  input=$PROJECT_DIR/en/data_text_normalization/test_cases_math.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
