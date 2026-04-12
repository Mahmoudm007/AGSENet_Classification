import re
from typing import Dict, List


CLASS_DESCRIPTIONS: Dict[str, str] = {
    "bare": (
        "The Bare class represents road surface conditions in which the traveled "
        "lane area immediately ahead of the vehicle is effectively free of snow "
        "accumulation within the tire-path region of interest. Images assigned "
        "to this class should show exposed pavement where both tire paths are "
        "clear, and there is no meaningful snow coverage affecting the lane "
        "segment the vehicle is about to drive on. Minor whitening, brightness, "
        "or visual texture that results from dry pavement, lighting, or surface "
        "reflectance should not be mistaken for snow. Likewise, snow located "
        "outside the traveled lane, such as on shoulders or emergency lanes, "
        "does not change the classification if the relevant tire-path area "
        "remains snow-free. This class therefore captures scenes where the "
        "driving surface is operationally clear and snow is not materially "
        "present in the vehicle's immediate path."
    ),
    "centre partly": (
        "The Centre Partly class describes roadway images in which both tire "
        "paths are largely exposed, but snow remains adjacent to the traveled "
        "wheel paths or around the central portion of the lane in a way that "
        "indicates partial snow presence within the lane environment. Images in "
        "this class should show a mostly drivable and visible pavement surface "
        "where the vehicle's immediate tire tracks are not fully obstructed by "
        "snow, yet snow is still clearly present just outside the main wheel "
        "paths or around the lane center. This class is appropriate when the "
        "road appears partially cleared or partially melted, and the snow "
        "pattern suggests that the lane is not fully bare but also not "
        "dominated by snow over the tire tracks. In practical terms, this class "
        "reflects a transitional condition where pavement exposure is "
        "substantial, but residual snow remains sufficiently visible within the "
        "lane context to distinguish it from a completely bare surface."
    ),
    "two track partly": (
        "The Two Track Partly class refers to conditions where both tire paths "
        "are visibly exposed, but snow remains between them, producing the "
        "characteristic appearance of two distinct wheel tracks through a "
        "snow-covered lane. Images in this category should clearly show that the "
        "vehicle would travel over two recognizable exposed tracks, with snow "
        "accumulation persisting in the center area between the tracks and often "
        "elsewhere in the lane as well. This is a classic partially snow-covered "
        "roadway condition in which traffic has created dual wheel paths through "
        "the snowpack, but the road cannot be considered mostly bare. The "
        "defining visual feature is the simultaneous visibility of both tire "
        "paths, rather than only one, and the continuing presence of snow in the "
        "lane surface between them. This class is especially important because "
        "it captures a common intermediate road state where traffic has begun to "
        "clear the lane but snow still materially influences the surface "
        "condition."
    ),
    "one track partly": (
        "The One Track Partly class is used when only one tire path is clearly "
        "exposed while the other remains snow-covered, creating an unbalanced "
        "and asymmetric snow distribution within the vehicle's immediate path. "
        "Images assigned to this class should show a lane where one side of the "
        "traveled path appears visible or nearly clear, while the corresponding "
        "opposite tire path still contains noticeable snow accumulation. The key "
        "distinction is that the two wheel paths are not equally exposed; rather, "
        "one track is available and the other is substantially obscured by snow. "
        "This class should therefore be used when the roadway surface condition "
        "immediately ahead of the vehicle suggests partial clearing or melting "
        "on only one side of the wheel path pattern. It captures a distinctly "
        "uneven road surface state that is visually and operationally different "
        "from both the dual-track partial condition and a fully snow-covered "
        "lane."
    ),
    "fully": (
        "The Fully class represents roadway images in which the two tire paths "
        "the vehicle is about to traverse are both covered with snow, even if "
        "some surrounding pavement or distant parts of the roadway appear "
        "visible elsewhere in the scene. Images in this class should show that "
        "the immediate drivable path is snow-covered across both wheel paths, "
        "indicating that the snow condition fully occupies the vehicle's intended "
        "travel area. Localized exposed pavement outside the relevant tire-path "
        "region does not override this classification if the actual wheel paths "
        "remain snow-covered. This class therefore corresponds to the highest "
        "level of snow presence among the defined roadway-condition categories "
        "and should be used for scenes where the traveled lane ahead is visually "
        "dominated by snow in the zone most relevant to vehicle movement."
    ),
}


DISPLAY_NAMES: Dict[str, str] = {
    "bare": "Bare",
    "centre partly": "Centre Partly",
    "two track partly": "Two Track Partly",
    "one track partly": "One Track Partly",
    "fully": "Fully",
}


ALIASES: Dict[str, str] = {
    "0 bare": "bare",
    "bare": "bare",
    "1 centre partly": "centre partly",
    "1 centre  partly": "centre partly",
    "1 centre - partly": "centre partly",
    "centre partly": "centre partly",
    "centre - partly": "centre partly",
    "2 two track partly": "two track partly",
    "2 two track - partly": "two track partly",
    "two track partly": "two track partly",
    "two track - partly": "two track partly",
    "3 one track partly": "one track partly",
    "3 one track - partly": "one track partly",
    "one track partly": "one track partly",
    "one track - partly": "one track partly",
    "4 fully": "fully",
    "fully": "fully",
}


def canonicalize_class_name(class_name: str) -> str:
    normalized = re.sub(r"[_-]+", " ", str(class_name).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    return ALIASES.get(normalized, normalized)


def get_class_description(class_name: str) -> str:
    key = canonicalize_class_name(class_name)
    if key not in CLASS_DESCRIPTIONS:
        raise KeyError(f"No description available for class '{class_name}'")
    return CLASS_DESCRIPTIONS[key]


def get_class_display_name(class_name: str) -> str:
    key = canonicalize_class_name(class_name)
    return DISPLAY_NAMES.get(key, str(class_name))


def get_descriptions_for_classes(class_names: List[str]) -> List[str]:
    return [get_class_description(name) for name in class_names]


def get_display_names(class_names: List[str]) -> List[str]:
    return [get_class_display_name(name) for name in class_names]

