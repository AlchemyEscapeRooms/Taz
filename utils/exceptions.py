"""
Custom Exception Hierarchy for AI Trading Bot

Provides specific, actionable exceptions instead of generic errors.
Makes debugging easier and enables precise error handling.

Exception Categories:
- Data Exceptions: Invalid or corrupt data
- API Exceptions: External service errors
- Configuration Exceptions: Config problems
- Trading Exceptions: Trading logic errors
- Security Exceptions: Security violations
- Validation Exceptions: Input validation failures
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for logging and alerting"""
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


class TradingBotException(Exception):
    """
    Base exception for all trading bot errors.

    All custom exceptions inherit from this, allowing catch-all
    handling while still maintaining specific error types.
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None
    ):
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.recovery_suggestion = recovery_suggestion
        super().__init__(self.formatted_message)

    @property
    def formatted_message(self) -> str:
        """Format error message with details and recovery suggestion"""
        msg = f"[{self.severity.name}] {self.message}"

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            msg += f" | Details: {details_str}"

        if self.recovery_suggestion:
            msg += f" | Suggestion: {self.recovery_suggestion}"

        return msg


# ============================================================================
# DATA EXCEPTIONS
# ============================================================================

class DataError(TradingBotException):
    """Base exception for data-related errors"""
    pass


class DataNotFoundError(DataError):
    """Requested data does not exist"""

    def __init__(self, resource: str, identifier: str, **kwargs):
        super().__init__(
            message=f"Data not found: {resource} '{identifier}'",
            severity=ErrorSeverity.WARNING,
            details={'resource': resource, 'identifier': identifier},
            recovery_suggestion="Check if the resource exists or try a different identifier",
            **kwargs
        )


class DataCorruptionError(DataError):
    """Data is corrupted or invalid"""

    def __init__(self, resource: str, reason: str, **kwargs):
        super().__init__(
            message=f"Data corruption detected in {resource}: {reason}",
            severity=ErrorSeverity.CRITICAL,
            details={'resource': resource, 'reason': reason},
            recovery_suggestion="Regenerate or re-download the data",
            **kwargs
        )


class InvalidDataFormatError(DataError):
    """Data format is invalid or unexpected"""

    def __init__(self, expected: str, got: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Invalid data format{' for ' + field if field else ''}: expected {expected}, got {got}",
            severity=ErrorSeverity.ERROR,
            details={'expected': expected, 'got': got, 'field': field},
            recovery_suggestion="Check data source and format specifications",
            **kwargs
        )


class DataValidationError(DataError):
    """Data failed validation checks"""

    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        super().__init__(
            message=f"Validation failed for {field}: {reason}",
            severity=ErrorSeverity.ERROR,
            details={'field': field, 'value': str(value), 'reason': reason},
            recovery_suggestion="Correct the data and try again",
            **kwargs
        )


# ============================================================================
# API EXCEPTIONS
# ============================================================================

class APIError(TradingBotException):
    """Base exception for API-related errors"""
    pass


class APIConnectionError(APIError):
    """Cannot connect to API service"""

    def __init__(self, service: str, reason: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Failed to connect to {service}" + (f": {reason}" if reason else ""),
            severity=ErrorSeverity.ERROR,
            details={'service': service, 'reason': reason},
            recovery_suggestion="Check internet connection and service status",
            **kwargs
        )


class APIAuthenticationError(APIError):
    """API authentication failed"""

    def __init__(self, service: str, **kwargs):
        super().__init__(
            message=f"Authentication failed for {service}",
            severity=ErrorSeverity.CRITICAL,
            details={'service': service},
            recovery_suggestion="Verify API keys in .env file",
            **kwargs
        )


class APIRateLimitError(APIError):
    """API rate limit exceeded"""

    def __init__(self, service: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(
            message=f"Rate limit exceeded for {service}",
            severity=ErrorSeverity.WARNING,
            details={'service': service, 'retry_after_seconds': retry_after},
            recovery_suggestion=f"Wait {retry_after} seconds before retrying" if retry_after else "Wait and retry later",
            **kwargs
        )


class APIQuotaExceededError(APIError):
    """API quota exceeded"""

    def __init__(self, service: str, quota_type: str, **kwargs):
        super().__init__(
            message=f"{quota_type} quota exceeded for {service}",
            severity=ErrorSeverity.ERROR,
            details={'service': service, 'quota_type': quota_type},
            recovery_suggestion="Upgrade API plan or wait for quota reset",
            **kwargs
        )


class APIResponseError(APIError):
    """API returned an error response"""

    def __init__(self, service: str, status_code: int, message: str, **kwargs):
        super().__init__(
            message=f"{service} API error ({status_code}): {message}",
            severity=ErrorSeverity.ERROR,
            details={'service': service, 'status_code': status_code, 'api_message': message},
            recovery_suggestion="Check API documentation and request parameters",
            **kwargs
        )


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(TradingBotException):
    """Base exception for configuration errors"""
    pass


class ConfigNotFoundError(ConfigurationError):
    """Configuration file not found"""

    def __init__(self, config_file: str, **kwargs):
        super().__init__(
            message=f"Configuration file not found: {config_file}",
            severity=ErrorSeverity.CRITICAL,
            details={'config_file': config_file},
            recovery_suggestion="Create configuration file or check path",
            **kwargs
        )


class InvalidConfigError(ConfigurationError):
    """Configuration file is invalid"""

    def __init__(self, config_file: str, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid configuration in {config_file}: {reason}",
            severity=ErrorSeverity.CRITICAL,
            details={'config_file': config_file, 'reason': reason},
            recovery_suggestion="Fix configuration file syntax and values",
            **kwargs
        )


class MissingConfigValueError(ConfigurationError):
    """Required configuration value is missing"""

    def __init__(self, key: str, section: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Missing required configuration: {key}" + (f" in section {section}" if section else ""),
            severity=ErrorSeverity.ERROR,
            details={'key': key, 'section': section},
            recovery_suggestion=f"Add {key} to configuration file",
            **kwargs
        )


class InvalidConfigValueError(ConfigurationError):
    """Configuration value is invalid"""

    def __init__(self, key: str, value: Any, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid configuration value for {key}: {reason}",
            severity=ErrorSeverity.ERROR,
            details={'key': key, 'value': str(value), 'reason': reason},
            recovery_suggestion=f"Correct {key} in configuration file",
            **kwargs
        )


# ============================================================================
# TRADING EXCEPTIONS
# ============================================================================

class TradingError(TradingBotException):
    """Base exception for trading-related errors"""
    pass


class InsufficientCapitalError(TradingError):
    """Not enough capital for trade"""

    def __init__(self, required: float, available: float, **kwargs):
        super().__init__(
            message=f"Insufficient capital: need ${required:.2f}, have ${available:.2f}",
            severity=ErrorSeverity.WARNING,
            details={'required': required, 'available': available, 'shortfall': required - available},
            recovery_suggestion="Add more capital or reduce position size",
            **kwargs
        )


class InvalidOrderError(TradingError):
    """Order parameters are invalid"""

    def __init__(self, reason: str, order_details: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            message=f"Invalid order: {reason}",
            severity=ErrorSeverity.ERROR,
            details=order_details or {},
            recovery_suggestion="Review and correct order parameters",
            **kwargs
        )


class PositionNotFoundError(TradingError):
    """Position does not exist"""

    def __init__(self, symbol: str, **kwargs):
        super().__init__(
            message=f"No open position found for {symbol}",
            severity=ErrorSeverity.WARNING,
            details={'symbol': symbol},
            recovery_suggestion="Check position list or symbol",
            **kwargs
        )


class RiskLimitExceededError(TradingError):
    """Risk limits have been exceeded"""

    def __init__(self, limit_type: str, limit: float, current: float, **kwargs):
        super().__init__(
            message=f"{limit_type} limit exceeded: {current:.2f} > {limit:.2f}",
            severity=ErrorSeverity.CRITICAL,
            details={'limit_type': limit_type, 'limit': limit, 'current': current},
            recovery_suggestion="Wait for limits to reset or adjust risk parameters",
            **kwargs
        )


class MarketClosedError(TradingError):
    """Attempted trading when market is closed"""

    def __init__(self, **kwargs):
        super().__init__(
            message="Market is currently closed",
            severity=ErrorSeverity.WARNING,
            recovery_suggestion="Wait for market hours or use paper trading",
            **kwargs
        )


# ============================================================================
# SECURITY EXCEPTIONS
# ============================================================================

class SecurityError(TradingBotException):
    """Base exception for security violations"""
    pass


class SecurityViolationError(SecurityError):
    """Security policy violation detected"""

    def __init__(self, violation_type: str, details: str, **kwargs):
        super().__init__(
            message=f"Security violation: {violation_type} - {details}",
            severity=ErrorSeverity.CRITICAL,
            details={'violation_type': violation_type, 'details': details},
            recovery_suggestion="This may indicate a security threat. Review input data.",
            **kwargs
        )


class SQLInjectionError(SecurityError):
    """SQL injection attempt detected"""

    def __init__(self, input_value: str, **kwargs):
        super().__init__(
            message="Potential SQL injection detected",
            severity=ErrorSeverity.CRITICAL,
            details={'attempted_input': input_value[:100]},  # Truncate for safety
            recovery_suggestion="Do not use special SQL characters in inputs",
            **kwargs
        )


class PathTraversalError(SecurityError):
    """Path traversal attempt detected"""

    def __init__(self, attempted_path: str, **kwargs):
        super().__init__(
            message="Path traversal attempt detected",
            severity=ErrorSeverity.CRITICAL,
            details={'attempted_path': attempted_path},
            recovery_suggestion="Use valid file paths only",
            **kwargs
        )


class UnauthorizedAccessError(SecurityError):
    """Unauthorized access attempt"""

    def __init__(self, resource: str, **kwargs):
        super().__init__(
            message=f"Unauthorized access attempt to {resource}",
            severity=ErrorSeverity.CRITICAL,
            details={'resource': resource},
            recovery_suggestion="Check permissions and authentication",
            **kwargs
        )


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(TradingBotException):
    """Base exception for validation errors"""
    pass


class InvalidInputError(ValidationError):
    """Input validation failed"""

    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid input for {field}: {reason}",
            severity=ErrorSeverity.ERROR,
            details={'field': field, 'value': str(value)[:100], 'reason': reason},
            recovery_suggestion="Provide valid input according to requirements",
            **kwargs
        )


class SchemaValidationError(ValidationError):
    """Data doesn't match expected schema"""

    def __init__(self, schema_name: str, errors: list, **kwargs):
        super().__init__(
            message=f"Schema validation failed for {schema_name}",
            severity=ErrorSeverity.ERROR,
            details={'schema': schema_name, 'errors': errors},
            recovery_suggestion="Ensure data matches schema requirements",
            **kwargs
        )


# ============================================================================
# RESOURCE EXCEPTIONS
# ============================================================================

class ResourceError(TradingBotException):
    """Base exception for resource management errors"""
    pass


class ResourceExhaustedError(ResourceError):
    """System resources exhausted"""

    def __init__(self, resource_type: str, **kwargs):
        super().__init__(
            message=f"{resource_type} resources exhausted",
            severity=ErrorSeverity.CRITICAL,
            details={'resource_type': resource_type},
            recovery_suggestion="Free up resources or increase limits",
            **kwargs
        )


class ConnectionPoolExhaustedError(ResourceError):
    """Database connection pool exhausted"""

    def __init__(self, pool_size: int, **kwargs):
        super().__init__(
            message=f"Connection pool exhausted (size: {pool_size})",
            severity=ErrorSeverity.ERROR,
            details={'pool_size': pool_size},
            recovery_suggestion="Increase pool size or check for connection leaks",
            **kwargs
        )


# ============================================================================
# STRATEGY EXCEPTIONS
# ============================================================================

class StrategyError(TradingBotException):
    """Base exception for strategy errors"""
    pass


class StrategyNotFoundError(StrategyError):
    """Strategy does not exist"""

    def __init__(self, strategy_name: str, available: Optional[list] = None, **kwargs):
        super().__init__(
            message=f"Strategy '{strategy_name}' not found",
            severity=ErrorSeverity.ERROR,
            details={'strategy_name': strategy_name, 'available_strategies': available},
            recovery_suggestion=f"Use one of: {', '.join(available)}" if available else "Check strategy name",
            **kwargs
        )


class StrategyExecutionError(StrategyError):
    """Strategy execution failed"""

    def __init__(self, strategy_name: str, reason: str, **kwargs):
        super().__init__(
            message=f"Strategy '{strategy_name}' execution failed: {reason}",
            severity=ErrorSeverity.ERROR,
            details={'strategy_name': strategy_name, 'reason': reason},
            recovery_suggestion="Check strategy parameters and data availability",
            **kwargs
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def handle_exception(exc: Exception, logger=None) -> ErrorSeverity:
    """
    Central exception handler that logs and returns severity.

    Args:
        exc: The exception to handle
        logger: Optional logger instance

    Returns:
        Error severity level
    """
    if isinstance(exc, TradingBotException):
        severity = exc.severity
        message = exc.formatted_message

        if logger:
            if severity == ErrorSeverity.CRITICAL:
                logger.critical(message)
            elif severity == ErrorSeverity.ERROR:
                logger.error(message)
            elif severity == ErrorSeverity.WARNING:
                logger.warning(message)
            else:
                logger.info(message)

        return severity
    else:
        # Generic exception
        if logger:
            logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
        return ErrorSeverity.ERROR


if __name__ == '__main__':
    # Test exception handling
    print("Testing Exception Hierarchy...")
    print("=" * 80)

    # Test various exceptions
    exceptions_to_test = [
        DataNotFoundError("stock_data", "AAPL"),
        APIAuthenticationError("Alpaca"),
        InsufficientCapitalError(10000, 5000),
        SQLInjectionError("'; DROP TABLE users--"),
        InvalidInputError("symbol", "123INVALID", "Must be letters only"),
        RiskLimitExceededError("daily_loss", 5.0, 7.5),
    ]

    for exc in exceptions_to_test:
        print(f"\n{type(exc).__name__}:")
        print(f"  Message: {exc.message}")
        print(f"  Severity: {exc.severity.name}")
        print(f"  Formatted: {exc.formatted_message}")

    print("\n" + "=" * 80)
    print("Exception tests complete!")
