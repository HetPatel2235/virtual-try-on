"""Size Recommendation Module"""

from typing import Dict, List
from src.models import Measurements, SizeRecommendation


def recommend_size(
    measurements: Measurements,
    size_chart: Dict[str, Dict[str, float]]
) -> SizeRecommendation:
    """Recommend best size based on measurements"""
    if not size_chart:
        raise ValueError("Size chart cannot be empty")
    
    fit_scores = {}
    for size, size_measurements in size_chart.items():
        score = calculate_fit_score(measurements, size_measurements)
        fit_scores[size] = score
    
    best_size = max(fit_scores, key=fit_scores.get)
    best_score = fit_scores[best_size]
    
    recommended_sizes = [
        size for size, score in fit_scores.items()
        if score > 0.7
    ]
    recommended_sizes.sort(key=lambda s: fit_scores[s], reverse=True)
    
    confidence = min(best_score, 0.95)
    
    return SizeRecommendation(
        size=best_size,
        confidence=confidence,
        fit_scores=fit_scores,
        recommended_sizes=recommended_sizes
    )


def calculate_fit_score(
    user_measurements: Measurements,
    size_measurements: Dict[str, float]
) -> float:
    """Calculate how well a size fits the user"""
    total_score = 0.0
    total_weight = 0.0
    
    if 'shoulder_width_cm' in size_measurements:
        shoulder_score = calculate_measurement_fit(
            user_measurements.shoulder_width_cm,
            size_measurements['shoulder_width_cm'],
            tolerance_percent=5
        )
        weight = 0.5
        total_score += shoulder_score * weight
        total_weight += weight
    
    if 'chest_circumference_cm' in size_measurements:
        chest_score = calculate_measurement_fit(
            user_measurements.chest_circumference_cm,
            size_measurements['chest_circumference_cm'],
            tolerance_percent=7
        )
        weight = 0.35
        total_score += chest_score * weight
        total_weight += weight
    
    if 'torso_length_cm' in size_measurements:
        torso_score = calculate_measurement_fit(
            user_measurements.torso_length_cm,
            size_measurements['torso_length_cm'],
            tolerance_percent=5
        )
        weight = 0.15
        total_score += torso_score * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_score / total_weight


def calculate_measurement_fit(
    user_value: float,
    size_value: float,
    tolerance_percent: float = 5.0
) -> float:
    """Calculate fit score for a single measurement"""
    if size_value == 0:
        return 0.0
    
    difference_percent = abs(user_value - size_value) / size_value * 100
    
    if difference_percent == 0:
        return 1.0
    
    max_difference = tolerance_percent * 2
    
    if difference_percent >= max_difference:
        return 0.0
    
    score = 1.0 - (difference_percent / max_difference)
    
    return max(0.0, min(1.0, score))


def get_size_alternatives(
    fit_scores: Dict[str, float],
    top_n: int = 3
) -> List[str]:
    """Get alternative size recommendations"""
    sorted_sizes = sorted(fit_scores.items(), key=lambda x: x[1], reverse=True)
    alternatives = [size for size, score in sorted_sizes[:top_n]]
    return alternatives


def explain_recommendation(
    recommendation: SizeRecommendation,
    measurements: Measurements
) -> str:
    """Generate human-readable explanation of recommendation"""
    lines = []
    lines.append(f"📏 Size Recommendation: {recommendation.size}")
    lines.append(f"✅ Confidence: {recommendation.confidence * 100:.1f}%")
    
    if recommendation.recommended_sizes:
        lines.append(f"💡 Also consider: {', '.join(recommendation.recommended_sizes[:2])}")
    
    lines.append(f"\n📐 Your Measurements:")
    lines.append(f"  • Shoulder width: {measurements.shoulder_width_cm:.1f} cm")
    lines.append(f"  • Chest circumference: {measurements.chest_circumference_cm:.1f} cm")
    lines.append(f"  • Torso length: {measurements.torso_length_cm:.1f} cm")
    
    if recommendation.confidence >= 0.85:
        lines.append(f"\n✨ This size should fit you well!")
    elif recommendation.confidence >= 0.70:
        lines.append(f"\n👍 This size is a good fit.")
    else:
        lines.append(f"\n⚠️  This size may not fit perfectly.")
    
    return "\n".join(lines)


def compare_sizes(
    size1: str,
    size2: str,
    size_chart: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Compare two sizes side by side"""
    if size1 not in size_chart or size2 not in size_chart:
        raise ValueError(f"Size not in chart: {size1} or {size2}")
    
    measurements1 = size_chart[size1]
    measurements2 = size_chart[size2]
    
    differences = {}
    for measurement in measurements1.keys():
        if measurement in measurements2:
            diff = measurements2[measurement] - measurements1[measurement]
            differences[measurement] = diff
    
    return differences


def find_closest_size(
    user_measurements: Measurements,
    size_chart: Dict[str, Dict[str, float]]
) -> str:
    """Find the single closest size"""
    fit_scores = {}
    for size, size_measurements in size_chart.items():
        score = calculate_fit_score(user_measurements, size_measurements)
        fit_scores[size] = score
    
    return max(fit_scores, key=fit_scores.get)