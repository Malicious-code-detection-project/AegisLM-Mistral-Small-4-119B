OLD_ID="g2-bf16-lora-fsdp2-20260818-001"
NEW_ID="g3-bf16-lora-fsdp2-20260819-001"

SOURCE_CONFIG="configs/g2-bf16-lora-fsdp2-10step-20260818-001.yaml"
G3_CONFIG="configs/g3-bf16-lora-fsdp2-100step-20260819-001.yaml"

test -f "$SOURCE_CONFIG"
test ! -e "$G3_CONFIG"

sed \
  -e "s/${OLD_ID}/${NEW_ID}/g" \
  -e 's/^max_steps: 10$/max_steps: 100/' \
  -e 's/^save_steps: 10$/save_steps: 100/' \
  "$SOURCE_CONFIG" > "$G3_CONFIG"

grep -nE \
  '^(dataset_prepared_path|output_dir|max_steps|logging_steps|save_strategy|save_steps):' \
  "$G3_CONFIG"

sha256sum "$G3_CONFIG"