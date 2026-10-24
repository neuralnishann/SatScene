"""Static domain-knowledge table used by the agent's *reasoning* step.

Keys are the **exact 14 class names** of the dataset (see ``data.yaml``).  Each
value is a ``(title, suggested_follow_up)`` pair.  The follow-ups are
*suggestions* for a human analyst -- the agent does not execute them.

NOTE: the original notebook keyed this table on strings such as ``"agricultural"``,
``"terminal"``, ``"car parking"``, ``"road"``, ``"train"`` and ``"domestic"``,
none of which match the dataset's class names (``Agriculture``, ``Airport``,
``Parking``, ``Highway``, ``Railway``); six of the fourteen classes (including
``Grassland``, which had no entry at all) therefore fell through to the
"Unknown Land Use" fallback.  This table fixes that.
"""
from __future__ import annotations

from typing import Dict, Tuple

KNOWLEDGE: Dict[str, Tuple[str, str]] = {
    "Agriculture": ("Agricultural land", "Compute NDVI time series for crop-health and phenology monitoring."),
    "Airport": ("Airport / terminal area", "Cross-reference flight-traffic data to gauge economic activity."),
    "Beach": ("Coastal beach", "Monitor shoreline change / erosion risk and seasonal tourism density."),
    "City": ("Urban city centre", "Analyse urban heat island effect, built-up growth and traffic patterns."),
    "Desert": ("Desert", "Assess solar-farm siting potential and dust-storm exposure."),
    "Forest": ("Forest", "Estimate canopy cover / carbon stock and watch for deforestation alerts."),
    "Grassland": ("Grassland / pasture", "Track vegetation seasonality (NDVI) and grazing pressure."),
    "Highway": ("Highway / road network", "Evaluate infrastructure condition and maintenance needs."),
    "Lake": ("Lake", "Monitor water-surface temperature, extent and algal-bloom risk."),
    "Mountain": ("Mountain terrain", "Assess slope stability and landslide susceptibility (DEM derivatives)."),
    "Parking": ("Parking area", "Estimate commercial activity from occupancy over time."),
    "Port": ("Seaport", "Track vessel and container throughput for supply-chain indicators."),
    "Railway": ("Railway / station", "Analyse commuter and freight transit loads."),
    "River": ("River", "Monitor flood-risk indicators and water quality."),
}

UNKNOWN = ("Unknown land use", "No domain rule is registered for this class; manual review recommended.")


def lookup(class_name: str) -> Tuple[str, str]:
    """Case-insensitive lookup with an explicit fallback."""
    for key, val in KNOWLEDGE.items():
        if key.lower() == class_name.lower():
            return val
    return UNKNOWN
