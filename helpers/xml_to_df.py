import os
import glob
import xml.etree.ElementTree as ET

import pandas as pd


def xml_to_df(annotations_path, images_dir) -> pd.DataFrame:
    """Takes images the source and the path to xml file, that stores classes annotations of those images,
     and returns a Pandas DataFrame that holds images info (dimensions, coordinates, etc)"""

    curr_dir = os.getcwd()

    xml_list = []

    for xml_file in glob.glob(annotations_path + "/*.xml"):
        tree = ET.parse(xml_file)
        images_data = tree.iter("image")

        files = os.listdir(os.path.join(curr_dir, os.pardir, images_dir))

        for idxx, file in enumerate(files):
            for idx, image_data in enumerate(images_data):
                name = image_data.attrib["name"]

                if file == name:
                    print(idxx, file)

                    try:
                        width = image_data.attrib["width"]
                        height = image_data.attrib["height"]

                        for box in image_data.findall("box"):
                            xbr = int(float(box.attrib["xbr"]))
                            xtl = int(float(box.attrib["xtl"]))
                            ybr = int(float(box.attrib["ybr"]))
                            ytl = int(float(box.attrib["ytl"]))

                            if xbr < xtl:
                                xmax = xtl
                                xmin = xbr
                            else:
                                xmin = xtl
                                xmax = xbr

                            if ybr < ytl:
                                ymax = ytl
                                ymin = ybr
                            else:
                                ymin = ytl
                                ymax = ybr

                            value = (
                                name,
                                width,
                                height,
                                box.attrib["label"],
                                int(xmin),
                                int(ymin),
                                int(xmax),
                                int(ymax),
                            )

                            xml_list.append(value)
                    except AttributeError as e:
                        print(f"Error: {e}")
                    break

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    return pd.DataFrame(xml_list, columns=column_name)
