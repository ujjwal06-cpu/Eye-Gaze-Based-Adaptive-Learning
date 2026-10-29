"""
Integration -- AOI to Intent Grid Mapping
=============================================
THE MISSING PIECE: none of the three assignment docs own "which AAC
grid cell does this fixation correspond to". Diya's gaze-processing
pipeline already computes an Area-Of-Interest (AOI) per fixation
(center/left/right/up/down), but those are just 5 directional screen
zones from her Week 2 synthetic generator -- they were never designed
to represent your 10 communication intents.

HONEST LIMITATION: your system needs 10 grid cells (one per intent),
but Diya's generator only ever simulates gaze landing on 5 target
locations. We cannot demo intents whose target location doesn't exist
in the data -- that would mean inventing fixations that never
happened. So this mapping covers 5 of your 10 intents for now, chosen
as plausible example placements. The honest fix is for Diya to extend
her ZONES dict (in generate_gaze_dataset.py) from 5 to 10 target
locations -- a small, mechanical change to a dict literal, not a
redesign -- so each of the 10 intents gets its own real screen target.

This mapping is a placeholder for exactly that reason: swap it for
whatever layout the team actually designs for the physical/on-screen
AAC grid.
"""

from intents import IntentName

# Placeholder 5-of-10 mapping -- replace once Diya's grid has 10 zones.
AOI_TO_INTENT: dict[str, IntentName] = {
    "center": IntentName.GENERAL_MESSAGE,
    "left": IntentName.WATER,
    "right": IntentName.HELP,
    "up": IntentName.MEDICINE,
    "down": IntentName.TOILET,
}

UNMAPPED_INTENTS = [
    i for i in IntentName if i not in AOI_TO_INTENT.values()
]  # FOOD, CALL, EMERGENCY, PAIN, ENTERTAINMENT -- need new grid zones


def aoi_to_intent(aoi: str) -> IntentName | None:
    """Map a Member-1 AOI label to a Member-3 intent, or None if unmapped."""
    return AOI_TO_INTENT.get(aoi)


if __name__ == "__main__":
    print("Currently mapped (5/10):")
    for aoi, intent in AOI_TO_INTENT.items():
        print(f"  {aoi:<8} -> {intent.value}")
    print("\nStill need real grid zones for:")
    for intent in UNMAPPED_INTENTS:
        print(f"  {intent.value}")
