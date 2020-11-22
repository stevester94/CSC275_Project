set -eou pipefail

for zip in $(find . |  grep zipped); do
    unzip $zip
    rm $zip
done

sha512sum * > SHA512

for meta in $(find . | grep meta); do
    sha512=$(jq -r '._metadata.global["core:sha512"]' $meta)
    transmitter_id=$(jq -r '._metadata.annotations[0]["wines:transmitter"].ID["Transmitter ID"]' $meta)
    transmission_id=$(jq -r '._metadata.annotations[0]["wines:transmitter"].ID["Transmission ID"]' $meta)

    bin_name=$(grep $sha512 SHA512 | awk '{print $2}')

    echo "----------"
    echo $sha512
    echo $transmitter_id
    echo $transmission_id
    echo $bin_name

    mkdir -p "Device_$transmitter_id/tx_$transmission_id/"
    mv $bin_name "Device_$transmitter_id/tx_$transmission_id/"
    mv $meta "Device_$transmitter_id/tx_$transmission_id/"

done
