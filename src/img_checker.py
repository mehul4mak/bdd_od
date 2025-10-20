def check_images(all_train_images, IMAGE_DIR_TRAIN, RESULTS_DIR) -> None:
    corrupted_files = 0
    duplicate_count = 0
    missing_files = 0
    hash_dict = defaultdict(list)

    for img_name in all_train_images:
        img_path = os.path.join(IMAGE_DIR_TRAIN, img_name)

        if not os.path.exists(img_path):
            print(f"Missing: {img_name}")
            missing_files += 1
            continue

        try:
            with Image.open(img_path) as img:
                img.load()
                img_hash = str(imagehash.average_hash(img))
                hash_dict[img_hash].append(img_name)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Corrupted: {img_name} ({e})")
            corrupted_files += 1

    # Now analyze duplicates
    duplicates = {h: names for h, names in hash_dict.items() if len(names) > 1}
    duplicate_count = sum(len(v) - 1 for v in duplicates.values())

    print("\n Summary:")
    print(f"Missing Files: {missing_files}")
    print(f"Corrupted Files: {corrupted_files}")
    print(f"Duplicate Groups: {len(duplicates)}")
    print(f"Duplicate Images (total extra files): {duplicate_count}")

    #  Show sample duplicate sets
    print("\n Example Duplicate Groups:")
    for i, (h, names) in enumerate(duplicates.items()):
        print(f"\nGroup {i + 1} (hash={h}):")
        for n in names:
            print(f"  - {n}")
        if i >= 4:  # show first 5 groups only
            break

    # Number of duplicate groups to visualize
    num_groups_to_show = 5

    for i, (h, names) in enumerate(duplicates.items()):
        if i >= num_groups_to_show:
            break

        n_images = len(names)
        plt.figure(figsize=(3 * n_images, 3))
        plt.suptitle(f"Duplicate Group {i + 1} (hash={h})", fontsize=16)

        for j, img_name in enumerate(names):
            img_path = os.path.join(IMAGE_DIR_TRAIN, img_name)
            try:
                with Image.open(img_path) as img:
                    plt.subplot(1, n_images, j + 1)
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(img_name, fontsize=10)
            except Exception as e:
                print(f"Could not open {img_name}: {e}")

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"duplicate_group_{i + 1}.png"))
