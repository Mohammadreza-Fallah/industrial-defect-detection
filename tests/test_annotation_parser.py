from pathlib import Path

import pytest

from src.data.annotation_parser import parse_annotation, parse_all_annotations

SAMPLE_XML = """<annotation>
    <filename>scratches_1.jpg</filename>
    <size>
        <width>200</width>
        <height>200</height>
        <depth>1</depth>
    </size>
    <object>
        <name>scratches</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>26</xmin>
            <ymin>12</ymin>
            <xmax>43</xmax>
            <ymax>171</ymax>
        </bndbox>
    </object>
    <object>
        <name>scratches</name>
        <difficult>1</difficult>
        <bndbox>
            <xmin>8</xmin>
            <ymin>184</ymin>
            <xmax>17</xmax>
            <ymax>196</ymax>
        </bndbox>
    </object>
</annotation>
"""


@pytest.fixture
def sample_xml_file(tmp_path: Path) -> Path:
    """Writes SAMPLE_XML to a temp file so we don't depend on the real
    dataset being downloaded to run these tests."""
    xml_path = tmp_path / "scratches_1.xml"
    xml_path.write_text(SAMPLE_XML)
    return xml_path


def test_parse_annotation_basic_fields(sample_xml_file):
    result = parse_annotation(sample_xml_file)
    assert result["filename"] == "scratches_1.jpg"
    assert result["width"] == 200
    assert result["height"] == 200


def test_parse_annotation_objects(sample_xml_file):
    result = parse_annotation(sample_xml_file)
    assert len(result["objects"]) == 2

    first = result["objects"][0]
    assert first["name"] == "scratches"
    assert first["difficult"] == 0
    assert first["bbox"] == (26, 12, 43, 171)

    second = result["objects"][1]
    assert second["difficult"] == 1
    assert second["bbox"] == (8, 184, 17, 196)


def test_parse_all_annotations_finds_all_files(tmp_path: Path):
    # Write 3 sample XML files into a temp folder
    for i in range(3):
        (tmp_path / f"sample_{i}.xml").write_text(SAMPLE_XML)

    results = parse_all_annotations(str(tmp_path))
    assert len(results) == 3
    assert all(r["filename"] == "scratches_1.jpg" for r in results)


def test_parse_all_annotations_empty_folder(tmp_path: Path):
    results = parse_all_annotations(str(tmp_path))
    assert results == []
