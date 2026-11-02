"""Exercise archive layout checks without requiring the large real dataset."""

import io
import tarfile
import zipfile

import pytest

from pattern_recognition_coursework.data.import_nabirds import import_archive


CONTENTS = {
    "nabirds/classes.txt": "1 Birds\n2 Example species\n",
    "nabirds/hierarchy.txt": "2 1\n",
    "nabirds/images.txt": "one example/a.jpg\n",
    "nabirds/image_class_labels.txt": "one 2\n",
    "nabirds/images/example/a.jpg": "tiny image placeholder",
}


def _archive(path, contents):
    with zipfile.ZipFile(path, "w") as output:
        for name, data in contents.items():
            output.writestr(name, data)


def _tar_archive(path, contents):
    with tarfile.open(path, "w:gz") as output:
        for name, data in contents.items():
            payload = data.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))


def test_import_places_dataset_under_project_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "download.zip"
    _archive(archive, CONTENTS)
    result = import_archive(archive, tmp_path)
    assert result["images"] == 1
    assert result["classes"] == 2
    assert (tmp_path / "data/raw/nabirds/images/example/a.jpg").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        import_archive(archive, tmp_path)


def test_import_rejects_escape_and_incomplete_dataset(tmp_path):
    archive = tmp_path / "escape.zip"
    _archive(archive, {**CONTENTS, "../outside.txt": "bad"})
    with pytest.raises(ValueError, match="Unsafe archive path"):
        import_archive(archive, tmp_path)
    assert not (tmp_path / "outside.txt").exists()
    archive = tmp_path / "incomplete.zip"
    _archive(archive, {key: value for key, value in CONTENTS.items() if key != "nabirds/images/example/a.jpg"})
    with pytest.raises(ValueError, match="no NABirds images"):
        import_archive(archive, tmp_path)
    assert not (tmp_path / "data/raw/nabirds").exists()


def test_tar_gz_import_and_missing_indexed_image(tmp_path):
    archive = tmp_path / "nabirds.tar.gz"
    _tar_archive(archive, {**CONTENTS,
                           "nabirds/images.txt": "one example/a.jpg\ntwo example/b.jpg\n",
                           "nabirds/image_class_labels.txt": "one 2\ntwo 2\n"})
    with pytest.raises(ValueError, match="Missing 1 indexed images"):
        import_archive(archive, tmp_path)
    assert not (tmp_path / "data/raw/nabirds").exists()
    _tar_archive(archive, CONTENTS)
    assert import_archive(archive, tmp_path)["images"] == 1


def test_tar_gz_rejects_links(tmp_path):
    archive = tmp_path / "linked.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in CONTENTS.items():
            payload = data.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("nabirds/images/example/link.jpg")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside.txt"
        output.addfile(link)
    with pytest.raises(ValueError, match="link or special file"):
        import_archive(archive, tmp_path)


def test_import_rejects_extracted_feature_directory(tmp_path):
    directory = tmp_path / "NABirds"
    directory.mkdir()
    with pytest.raises(ValueError, match="archive file"):
        import_archive(directory, tmp_path)
