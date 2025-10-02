from huggingface_hub import snapshot_download


def main():
    snapshot_download(
        repo_id='dzimmerdkfz/mood_abdomen',
        repo_type='dataset',
        etag_timeout=60,
        max_workers=16
    )


if __name__ == '__main__':
    main()
