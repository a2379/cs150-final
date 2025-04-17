csv_file="*.csv"
delimiter=","
while IFS="$delimiter" read -r -a fields; do
  url="${fields[0]}"
  id="${url##*-}.midi"
  curl -o "$id" "$url"
  echo $id
done <"$csv_file"
