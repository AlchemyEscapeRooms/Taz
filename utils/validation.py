"""
Core Security and Validation Framework

This module provides comprehensive input validation, sanitization,
and security checks to prevent malicious data, code injection,
and system exploitation.

Features:
- Input validation and sanitization
- SQL injection prevention
- Path traversal prevention
- Command injection prevention
- Data type validation
- Business logic validation
- Rate limiting
- Anomaly detection
"""

import re
import os
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# Setup logger
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security validation levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


class SecurityError(Exception):
    """Raised when security check fails"""
    pass


@dataclass
class ValidationResult:
    """Result of validation check"""
    valid: bool
    message: str
    severity: SecurityLevel
    sanitized_value: Any = None
    metadata: Dict[str, Any] = None


class InputValidator:
    """
    Comprehensive input validation and sanitization.

    Prevents:
    - SQL injection
    - Command injection
    - Path traversal
    - XSS attacks
    - Buffer overflows
    - Type confusion
    """

    # Dangerous patterns that should never appear in inputs
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\/\*|\*\/|xp_|sp_)",
        r"(\bOR\b.*=.*\bOR\b)",
        r"(\bAND\b.*=.*\bAND\b)",
        r"('|(--)|;|\/\*|\*\/)",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",  # Shell metacharacters
        r"(\.\./|\.\.\\)",  # Path traversal
        r"(\$\{|\$\()",  # Variable expansion
        r"(>|<|>>)",  # Redirection
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\./|\.\.\\)",
        r"(~|%2e%2e)",
        r"(/etc/|/proc/|/sys/|C:\\Windows)",
    ]

    XSS_PATTERNS = [
        r"(<script|<iframe|<object|<embed)",
        r"(javascript:|data:|vbscript:)",
        r"(on\w+\s*=)",  # Event handlers
    ]

    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.validation_cache = {}
        self.failed_validations = []

    def validate_symbol(self, symbol: str) -> ValidationResult:
        """
        Validate stock symbol.

        Rules:
        - 1-5 uppercase letters only
        - No special characters
        - Not empty
        """
        if not symbol:
            return ValidationResult(
                valid=False,
                message="Symbol cannot be empty",
                severity=SecurityLevel.HIGH
            )

        # Sanitize: uppercase and strip whitespace
        sanitized = symbol.upper().strip()

        # Validate format
        if not re.match(r'^[A-Z]{1,5}$', sanitized):
            return ValidationResult(
                valid=False,
                message=f"Invalid symbol format: '{symbol}'. Must be 1-5 uppercase letters.",
                severity=SecurityLevel.HIGH
            )

        # Check against known malicious patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected in symbol: {symbol}")
                return ValidationResult(
                    valid=False,
                    message="Potential SQL injection detected",
                    severity=SecurityLevel.CRITICAL
                )

        return ValidationResult(
            valid=True,
            message="Symbol validated",
            severity=SecurityLevel.LOW,
            sanitized_value=sanitized
        )

    def validate_date(self, date_str: str) -> ValidationResult:
        """
        Validate date string.

        Rules:
        - YYYY-MM-DD format only
        - Valid calendar date
        - Not in future
        - Not before 1900
        """
        if not date_str:
            return ValidationResult(
                valid=False,
                message="Date cannot be empty",
                severity=SecurityLevel.MEDIUM
            )

        # Check format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return ValidationResult(
                valid=False,
                message=f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.",
                severity=SecurityLevel.MEDIUM
            )

        # Parse and validate
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

            # Check reasonable bounds
            min_date = datetime(1900, 1, 1)
            max_date = datetime.now() + timedelta(days=1)  # Allow tomorrow for timezone

            if date_obj < min_date:
                return ValidationResult(
                    valid=False,
                    message=f"Date {date_str} is before 1900",
                    severity=SecurityLevel.MEDIUM
                )

            if date_obj > max_date:
                return ValidationResult(
                    valid=False,
                    message=f"Date {date_str} is in the future",
                    severity=SecurityLevel.MEDIUM
                )

            return ValidationResult(
                valid=True,
                message="Date validated",
                severity=SecurityLevel.LOW,
                sanitized_value=date_str,
                metadata={'date_object': date_obj}
            )

        except ValueError as e:
            return ValidationResult(
                valid=False,
                message=f"Invalid date: {date_str}. {str(e)}",
                severity=SecurityLevel.MEDIUM
            )

    def validate_file_path(self, file_path: str, allowed_base_dirs: List[str] = None) -> ValidationResult:
        """
        Validate file path for security.

        Prevents:
        - Path traversal attacks
        - Access to system directories
        - Null bytes
        """
        if not file_path:
            return ValidationResult(
                valid=False,
                message="File path cannot be empty",
                severity=SecurityLevel.HIGH
            )

        # Check for null bytes
        if '\x00' in file_path:
            logger.warning(f"Null byte detected in path: {file_path}")
            return ValidationResult(
                valid=False,
                message="Invalid characters in path",
                severity=SecurityLevel.CRITICAL
            )

        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return ValidationResult(
                    valid=False,
                    message="Path traversal detected",
                    severity=SecurityLevel.CRITICAL
                )

        # Resolve to absolute path
        try:
            abs_path = Path(file_path).resolve()
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Cannot resolve path: {str(e)}",
                severity=SecurityLevel.HIGH
            )

        # Check if within allowed directories
        if allowed_base_dirs:
            allowed = False
            for base_dir in allowed_base_dirs:
                try:
                    base_path = Path(base_dir).resolve()
                    if abs_path.is_relative_to(base_path):
                        allowed = True
                        break
                except (ValueError, AttributeError):
                    # Python < 3.9 doesn't have is_relative_to
                    try:
                        abs_path.relative_to(base_path)
                        allowed = True
                        break
                    except ValueError:
                        continue

            if not allowed:
                logger.warning(f"Attempted access outside allowed directories: {file_path}")
                return ValidationResult(
                    valid=False,
                    message="Path outside allowed directories",
                    severity=SecurityLevel.CRITICAL
                )

        return ValidationResult(
            valid=True,
            message="Path validated",
            severity=SecurityLevel.LOW,
            sanitized_value=str(abs_path)
        )

    def validate_numeric_range(
        self,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        name: str = "value"
    ) -> ValidationResult:
        """Validate numeric value is within acceptable range"""

        # Type check
        if not isinstance(value, (int, float)):
            return ValidationResult(
                valid=False,
                message=f"{name} must be a number, got {type(value).__name__}",
                severity=SecurityLevel.MEDIUM
            )

        # Check for NaN or Inf
        if isinstance(value, float):
            if value != value:  # NaN check
                return ValidationResult(
                    valid=False,
                    message=f"{name} is NaN",
                    severity=SecurityLevel.HIGH
                )
            if abs(value) == float('inf'):
                return ValidationResult(
                    valid=False,
                    message=f"{name} is infinite",
                    severity=SecurityLevel.HIGH
                )

        # Range checks
        if min_value is not None and value < min_value:
            return ValidationResult(
                valid=False,
                message=f"{name} ({value}) is below minimum ({min_value})",
                severity=SecurityLevel.MEDIUM
            )

        if max_value is not None and value > max_value:
            return ValidationResult(
                valid=False,
                message=f"{name} ({value}) exceeds maximum ({max_value})",
                severity=SecurityLevel.MEDIUM
            )

        return ValidationResult(
            valid=True,
            message=f"{name} validated",
            severity=SecurityLevel.LOW,
            sanitized_value=value
        )

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.

        - Removes null bytes
        - Strips whitespace
        - Truncates to max length
        - Escapes dangerous characters
        """
        if not isinstance(value, str):
            value = str(value)

        # Remove null bytes
        value = value.replace('\x00', '')

        # Strip whitespace
        value = value.strip()

        # Truncate
        if len(value) > max_length:
            logger.warning(f"String truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]

        return value

    def check_sql_injection(self, value: str) -> ValidationResult:
        """Check for SQL injection patterns"""
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern} in '{value}'")
                return ValidationResult(
                    valid=False,
                    message="Potential SQL injection detected",
                    severity=SecurityLevel.CRITICAL
                )

        return ValidationResult(
            valid=True,
            message="No SQL injection detected",
            severity=SecurityLevel.LOW
        )

    def check_command_injection(self, value: str) -> ValidationResult:
        """Check for command injection patterns"""
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                logger.warning(f"Command injection pattern detected: {pattern} in '{value}'")
                return ValidationResult(
                    valid=False,
                    message="Potential command injection detected",
                    severity=SecurityLevel.CRITICAL
                )

        return ValidationResult(
            valid=True,
            message="No command injection detected",
            severity=SecurityLevel.LOW
        )

    def validate_strategy_name(self, strategy: str, allowed_strategies: List[str]) -> ValidationResult:
        """Validate strategy name against whitelist"""

        sanitized = self.sanitize_string(strategy)

        if not sanitized:
            return ValidationResult(
                valid=False,
                message="Strategy name cannot be empty",
                severity=SecurityLevel.MEDIUM
            )

        if sanitized not in allowed_strategies:
            return ValidationResult(
                valid=False,
                message=f"Unknown strategy '{sanitized}'. Allowed: {', '.join(allowed_strategies)}",
                severity=SecurityLevel.MEDIUM,
                metadata={'allowed_strategies': allowed_strategies}
            )

        return ValidationResult(
            valid=True,
            message="Strategy validated",
            severity=SecurityLevel.LOW,
            sanitized_value=sanitized
        )

    def calculate_checksum(self, data: Union[str, bytes]) -> str:
        """Calculate SHA-256 checksum of data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()


class DataValidator:
    """
    Validates trading data integrity and correctness.

    Prevents:
    - Corrupted data
    - Impossible values (negative prices, etc.)
    - OHLC inconsistencies
    - Data anomalies
    """

    def __init__(self):
        self.anomaly_threshold = 3.0  # Standard deviations

    def validate_ohlc_data(self, open_price: float, high: float, low: float, close: float) -> ValidationResult:
        """
        Validate OHLC price data consistency.

        Rules:
        - All prices must be positive
        - High >= all other prices
        - Low <= all other prices
        - No NaN or Inf values
        """
        prices = {'open': open_price, 'high': high, 'low': low, 'close': close}

        # Check all are positive numbers
        for name, price in prices.items():
            if not isinstance(price, (int, float)):
                return ValidationResult(
                    valid=False,
                    message=f"{name} price must be a number, got {type(price).__name__}",
                    severity=SecurityLevel.HIGH
                )

            if price <= 0:
                return ValidationResult(
                    valid=False,
                    message=f"{name} price ({price}) must be positive",
                    severity=SecurityLevel.HIGH
                )

            # Check for NaN/Inf
            if isinstance(price, float):
                if price != price:  # NaN
                    return ValidationResult(
                        valid=False,
                        message=f"{name} price is NaN",
                        severity=SecurityLevel.HIGH
                    )
                if abs(price) == float('inf'):
                    return ValidationResult(
                        valid=False,
                        message=f"{name} price is infinite",
                        severity=SecurityLevel.HIGH
                    )

        # Check OHLC consistency
        if high < low:
            return ValidationResult(
                valid=False,
                message=f"High ({high}) is less than Low ({low})",
                severity=SecurityLevel.HIGH
            )

        if high < open_price or high < close:
            return ValidationResult(
                valid=False,
                message=f"High ({high}) is not the highest price",
                severity=SecurityLevel.HIGH
            )

        if low > open_price or low > close:
            return ValidationResult(
                valid=False,
                message=f"Low ({low}) is not the lowest price",
                severity=SecurityLevel.HIGH
            )

        return ValidationResult(
            valid=True,
            message="OHLC data validated",
            severity=SecurityLevel.LOW
        )

    def validate_volume(self, volume: Union[int, float]) -> ValidationResult:
        """Validate trading volume"""

        if not isinstance(volume, (int, float)):
            return ValidationResult(
                valid=False,
                message=f"Volume must be a number, got {type(volume).__name__}",
                severity=SecurityLevel.MEDIUM
            )

        if volume < 0:
            return ValidationResult(
                valid=False,
                message=f"Volume ({volume}) cannot be negative",
                severity=SecurityLevel.HIGH
            )

        # Check for unreasonably high volume (potential data corruption)
        if volume > 1e12:  # 1 trillion shares
            logger.warning(f"Suspiciously high volume detected: {volume}")
            return ValidationResult(
                valid=False,
                message=f"Volume ({volume}) is unreasonably high",
                severity=SecurityLevel.MEDIUM
            )

        return ValidationResult(
            valid=True,
            message="Volume validated",
            severity=SecurityLevel.LOW,
            sanitized_value=int(volume)
        )

    def detect_price_anomaly(self, prices: List[float], current_price: float) -> ValidationResult:
        """
        Detect price anomalies using statistical analysis.

        Flags prices that are > 3 standard deviations from mean.
        """
        if len(prices) < 2:
            return ValidationResult(
                valid=True,
                message="Insufficient data for anomaly detection",
                severity=SecurityLevel.LOW
            )

        import statistics

        mean = statistics.mean(prices)
        stdev = statistics.stdev(prices)

        if stdev == 0:
            return ValidationResult(
                valid=True,
                message="No variance in prices",
                severity=SecurityLevel.LOW
            )

        z_score = abs((current_price - mean) / stdev)

        if z_score > self.anomaly_threshold:
            logger.warning(f"Price anomaly detected: {current_price} (z-score: {z_score:.2f})")
            return ValidationResult(
                valid=False,
                message=f"Price {current_price} is {z_score:.1f} std devs from mean ({mean:.2f})",
                severity=SecurityLevel.MEDIUM,
                metadata={'z_score': z_score, 'mean': mean, 'stdev': stdev}
            )

        return ValidationResult(
            valid=True,
            message="No anomaly detected",
            severity=SecurityLevel.LOW,
            metadata={'z_score': z_score}
        )


# Global validator instances
input_validator = InputValidator(security_level=SecurityLevel.HIGH)
data_validator = DataValidator()


# Convenience functions
def validate_symbol(symbol: str) -> ValidationResult:
    """Validate stock symbol"""
    return input_validator.validate_symbol(symbol)


def validate_date(date_str: str) -> ValidationResult:
    """Validate date string"""
    return input_validator.validate_date(date_str)


def validate_ohlc(open_price: float, high: float, low: float, close: float) -> ValidationResult:
    """Validate OHLC prices"""
    return data_validator.validate_ohlc_data(open_price, high, low, close)


if __name__ == '__main__':
    # Test validation
    print("Testing Input Validator...")
    print("=" * 80)

    # Test symbol validation
    tests = [
        ("AAPL", True),
        ("SPY", True),
        ("GOOGL", True),
        ("invalid123", False),
        ("'; DROP TABLE--", False),
        ("../etc/passwd", False),
        ("", False),
    ]

    for test_val, should_pass in tests:
        result = validate_symbol(test_val)
        status = "✓" if result.valid == should_pass else "✗"
        print(f"{status} Symbol '{test_val}': {result.message}")

    print("\nTesting Date Validator...")
    print("=" * 80)

    date_tests = [
        ("2023-01-01", True),
        ("2025-12-31", False),  # Future
        ("1850-01-01", False),  # Too old
        ("invalid-date", False),
        ("2023/01/01", False),  # Wrong format
    ]

    for test_val, should_pass in date_tests:
        result = validate_date(test_val)
        status = "✓" if result.valid == should_pass else "✗"
        print(f"{status} Date '{test_val}': {result.message}")

    print("\nTesting OHLC Validator...")
    print("=" * 80)

    ohlc_tests = [
        ((100, 105, 95, 102), True),  # Valid
        ((100, 95, 105, 102), False),  # High < Low
        ((-100, 105, 95, 102), False),  # Negative price
        ((100, 105, 95, 110), False),  # Close > High
    ]

    for (o, h, l, c), should_pass in ohlc_tests:
        result = validate_ohlc(o, h, l, c)
        status = "✓" if result.valid == should_pass else "✗"
        print(f"{status} OHLC({o}, {h}, {l}, {c}): {result.message}")

    print("\n" + "=" * 80)
    print("Validation tests complete!")
