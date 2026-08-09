"""
Parses NEU-DET Pascal-VOC-style XML annotation files into plain Python
dicts, for use in the Phase 3 detection pipeline.
"""

from __future__ import annotations

from pathlib import Path

import xml.etree.ElementTree as ET


def parse_annotation(xml_path: str) -> dict:
    """Parses a single NEU-DET annotation XML file.

    Args:
        xml_path: path to a .xml annotation file.

    Returns:
        A dict shaped like:
        {
            'filename': str,
            'width': int,
            'height': int,
            'objects': [
                {'name': str, 'bbox': (xmin, ymin, xmax, ymax), 'difficult': int},
                ...
            ]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    objects = []
    for obj in root.findall('object'):
        obj_dict = {
            'name': obj.find('name').text,
            'difficult': int(obj.find('difficult').text),
            'bbox': (
                int(obj.find('bndbox/xmin').text),
                int(obj.find('bndbox/ymin').text),
                int(obj.find('bndbox/xmax').text),
                int(obj.find('bndbox/ymax').text),
            ),
        }
        objects.append(obj_dict)

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def parse_all_annotations(annotations_dir: str) -> list[dict]:
    """Parses every .xml file in a folder using parse_annotation().

    EXERCISE — fill in the TODOs below yourself.
    Concepts used: functions, lists, for loops, working with paths
    (pathlib), calling a function you already wrote.

    Args:
        annotations_dir: path to a folder containing .xml files
            (e.g. 'data/raw/NEU-DET/train/annotations').

    Returns:
        A list of dicts — one per XML file, each shaped like the
        return value of parse_annotation().
    """
    all_annotations = []

    folder = Path(annotations_dir)
    xml_files = sorted(folder.glob('*.xml'))

    for xml_file in xml_files:
        result = parse_annotation(xml_file)
        all_annotations.append(result)

    return all_annotations


if __name__ == "__main__":
    # Quick manual test once you've filled in the TODOs above.
    sample_dir = "data/raw/NEU-DET/train/annotations"
    results = parse_all_annotations(sample_dir)
    print(f"Parsed {len(results)} annotation files.")
    if results:
        print("First result:", results[0])
