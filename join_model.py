import os
def join_files(output_file, parts_pattern="best_model.pt.part"):
    i = 0
    with open(output_file, 'wb') as outfile:
        while True:
            part_filename = f"{parts_pattern}{i:03d}"
            if not os.path.exists(part_filename):
                break
            with open(part_filename, 'rb') as part_file:
                outfile.write(part_file.read())
            i += 1
    print(f"Archivo reconstruido como {output_file}")
join_files("best_model.pt")
