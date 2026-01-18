"""Date calculator tool for calculating differences between two dates."""
from datetime import datetime
from langchain_core.tools import tool


@tool
def calculate_date_difference(
    date1: str,
    date2: str
) -> str:
    """
    Calculate the difference between two dates.
    
    This tool calculates the time difference between two dates and returns
    the result in days, weeks, months, and years.
    
    Args:
        date1: First date as a string in YYYY-MM-DD format (e.g., "2024-01-15")
        date2: Second date as a string in YYYY-MM-DD format (e.g., "2024-03-20")
    
    Returns:
        A string describing the difference between the two dates in various units.
    
    Examples:
        calculate_date_difference("2024-01-15", "2024-03-20")
    """
    DATE_FORMAT = "%Y-%m-%d"
    
    try:
        # Parse both dates
        dt1 = datetime.strptime(date1.strip(), DATE_FORMAT)
        dt2 = datetime.strptime(date2.strip(), DATE_FORMAT)
        
        # Calculate the absolute difference
        if dt1 > dt2:
            earlier_date = dt2
            later_date = dt1
            direction = "after"
        else:
            earlier_date = dt1
            later_date = dt2
            direction = "before"
        
        delta = later_date - earlier_date
        total_days = delta.days
        total_seconds = delta.total_seconds()
        
        # Calculate different units
        days = total_days
        weeks = days / 7
        months = days / 30.44  # Average days per month
        years = days / 365.25  # Account for leap years
        
        # Format the result
        result_parts = [
            f"Date difference: {days} day(s)",
            f"{weeks:.2f} week(s)",
            f"{months:.2f} month(s)",
            f"{years:.2f} year(s)"
        ]
        
        result = f"The second date ({dt2.strftime('%Y-%m-%d')}) is {direction} the first date ({dt1.strftime('%Y-%m-%d')}) by:\n"
        result += "\n".join(f"- {part}" for part in result_parts)
        
        # Add additional context
        if total_seconds < 86400:  # Less than a day
            hours = total_seconds / 3600
            minutes = (total_seconds % 3600) / 60
            result += f"\n- {hours:.1f} hour(s) ({minutes:.1f} minute(s))"
        return result
        
    except ValueError as e:
        return f"Error: {str(e)}. Please provide dates in YYYY-MM-DD format (e.g., '2024-01-15')."
    except Exception as e:
        return f"Unexpected error: {str(e)}"
