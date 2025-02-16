"""Debug utilities for joao."""
import os
from typing import Optional

def is_debug_enabled(component: Optional[str] = None) -> bool:
    """Check if debug is enabled for a component.
    
    Args:
        component: Optional component name to check. If None, checks if any debug is enabled.
        
    Returns:
        bool: True if debug is enabled for the component or any component if component is None
    """
    debug_env = os.getenv('DEBUG', '').lower()
    if not debug_env:
        return False
        
    components = [c.strip() for c in debug_env.split(',')]
    return not component or component in components

def debug_print(component: str, *args, **kwargs):
    """Print debug message if debug is enabled for component.
    
    Args:
        component: Component name to check debug status for
        *args: Arguments to pass to print
        **kwargs: Keyword arguments to pass to print
    """
    if is_debug_enabled(component):
        print(f"[DEBUG {component.upper()}]", *args, **kwargs)
