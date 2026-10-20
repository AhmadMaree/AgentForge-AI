### Build Custom Excptions 

class AgentForgeError(Exception):
    """Base excption for our framework"""
    ...

class LLMError(AgentForgeError):
    """Raised when a model request fail!"""
    ...

class ToolsError(AgentForgeError):
    """Raised whe tool can't excute"""
    ...

class ConfigurationError(AgentForgeError):
    """Raised when configration is invalid"""
    ...

class ToolValidtionError(ToolsError):
    """Tool input faild schema validtion"""
    ...

class ToolNotFoundError(ToolsError):
    """Request tool not registry"""
    ...

class ToolExcutionError(ToolsError):
    """Tool Excution faild"""
    ...