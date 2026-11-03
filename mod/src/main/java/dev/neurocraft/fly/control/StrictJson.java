package dev.neurocraft.fly.control;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Small dependency-free JSON codec with duplicate-key, depth, node, and UTF-8 byte bounds. */
final class StrictJson {
    private static final int MAX_DEPTH = 16;
    private static final int MAX_NODES = 1_024;

    private final String input;
    private int index;
    private int nodes;

    private StrictJson(String input) {
        this.input = input;
    }

    static Map<String, Object> parseObject(String input, int maxBytes) throws ProtocolException {
        if (input == null) {
            throw new ProtocolException("message is null");
        }
        if (input.getBytes(StandardCharsets.UTF_8).length > maxBytes) {
            throw new ProtocolException("message exceeds byte ceiling");
        }
        StrictJson parser = new StrictJson(input);
        Object value = parser.parseValue(0);
        parser.skipWhitespace();
        if (parser.index != input.length()) {
            throw parser.error("trailing content");
        }
        if (!(value instanceof Map<?, ?> raw)) {
            throw parser.error("top-level JSON value must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> object = (Map<String, Object>) raw;
        return object;
    }

    static String encode(Object value, int maxBytes) {
        StringBuilder output = new StringBuilder(1_024);
        encodeValue(output, value, 0);
        String result = output.toString();
        if (result.getBytes(StandardCharsets.UTF_8).length > maxBytes) {
            throw new IllegalArgumentException("encoded message exceeds byte ceiling");
        }
        return result;
    }

    private Object parseValue(int depth) throws ProtocolException {
        if (depth > MAX_DEPTH) {
            throw error("JSON nesting exceeds depth ceiling");
        }
        if (++nodes > MAX_NODES) {
            throw error("JSON value count exceeds node ceiling");
        }
        skipWhitespace();
        char current = peek();
        return switch (current) {
            case '{' -> parseObjectValue(depth + 1);
            case '[' -> parseArray(depth + 1);
            case '"' -> parseString();
            case 't' -> parseLiteral("true", Boolean.TRUE);
            case 'f' -> parseLiteral("false", Boolean.FALSE);
            case 'n' -> parseLiteral("null", null);
            default -> parseNumber();
        };
    }

    private Map<String, Object> parseObjectValue(int depth) throws ProtocolException {
        Map<String, Object> result = new LinkedHashMap<>();
        expect('{');
        skipWhitespace();
        if (take('}')) {
            return result;
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                throw error("object key must be a string");
            }
            String key = parseString();
            skipWhitespace();
            expect(':');
            Object value = parseValue(depth);
            if (result.containsKey(key)) {
                throw error("duplicate object key");
            }
            result.put(key, value);
            skipWhitespace();
            if (take('}')) {
                return result;
            }
            expect(',');
        }
    }

    private List<Object> parseArray(int depth) throws ProtocolException {
        List<Object> result = new ArrayList<>();
        expect('[');
        skipWhitespace();
        if (take(']')) {
            return result;
        }
        while (true) {
            result.add(parseValue(depth));
            skipWhitespace();
            if (take(']')) {
                return Collections.unmodifiableList(new ArrayList<>(result));
            }
            expect(',');
        }
    }

    private String parseString() throws ProtocolException {
        expect('"');
        StringBuilder value = new StringBuilder();
        while (index < input.length()) {
            char current = input.charAt(index++);
            if (current == '"') {
                validateSurrogates(value);
                return value.toString();
            }
            if (current < 0x20) {
                throw error("unescaped control character");
            }
            if (current != '\\') {
                value.append(current);
                continue;
            }
            if (index >= input.length()) {
                throw error("unterminated escape");
            }
            char escaped = input.charAt(index++);
            switch (escaped) {
                case '"', '\\', '/' -> value.append(escaped);
                case 'b' -> value.append('\b');
                case 'f' -> value.append('\f');
                case 'n' -> value.append('\n');
                case 'r' -> value.append('\r');
                case 't' -> value.append('\t');
                case 'u' -> value.append(parseUnicodeEscape());
                default -> throw error("unsupported string escape");
            }
        }
        throw error("unterminated string");
    }

    private char parseUnicodeEscape() throws ProtocolException {
        if (index + 4 > input.length()) {
            throw error("short unicode escape");
        }
        int code = 0;
        for (int offset = 0; offset < 4; offset++) {
            int digit = Character.digit(input.charAt(index++), 16);
            if (digit < 0) {
                throw error("invalid unicode escape");
            }
            code = (code << 4) | digit;
        }
        return (char) code;
    }

    private BigDecimal parseNumber() throws ProtocolException {
        int start = index;
        take('-');
        if (take('0')) {
            if (isDigit(peekOrZero())) {
                throw error("leading zero in number");
            }
        } else {
            requireDigitOneToNine();
            while (isDigit(peekOrZero())) {
                index++;
            }
        }
        if (take('.')) {
            requireDigit();
            while (isDigit(peekOrZero())) {
                index++;
            }
        }
        char next = peekOrZero();
        if (next == 'e' || next == 'E') {
            index++;
            next = peekOrZero();
            if (next == '+' || next == '-') {
                index++;
            }
            requireDigit();
            while (isDigit(peekOrZero())) {
                index++;
            }
        }
        if (index == start) {
            throw error("expected JSON value");
        }
        try {
            BigDecimal result = new BigDecimal(input.substring(start, index)).stripTrailingZeros();
            return result.signum() == 0 ? BigDecimal.ZERO : result;
        } catch (NumberFormatException exception) {
            throw new ProtocolException("invalid number at character " + start, exception);
        }
    }

    private Object parseLiteral(String literal, Object value) throws ProtocolException {
        if (!input.startsWith(literal, index)) {
            throw error("invalid JSON literal");
        }
        index += literal.length();
        return value;
    }

    private static void encodeValue(StringBuilder output, Object value, int depth) {
        if (depth > MAX_DEPTH) {
            throw new IllegalArgumentException("JSON nesting exceeds depth ceiling");
        }
        if (value == null) {
            output.append("null");
        } else if (value instanceof String string) {
            quote(output, string);
        } else if (value instanceof Boolean bool) {
            output.append(bool);
        } else if (value instanceof Byte || value instanceof Short || value instanceof Integer
                || value instanceof Long || value instanceof BigDecimal) {
            output.append(value);
        } else if (value instanceof Float number) {
            requireFinite(number.doubleValue());
            output.append(Float.toString(number));
        } else if (value instanceof Double number) {
            requireFinite(number);
            output.append(Double.toString(number));
        } else if (value instanceof Map<?, ?> object) {
            output.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : object.entrySet()) {
                if (!(entry.getKey() instanceof String key)) {
                    throw new IllegalArgumentException("JSON object keys must be strings");
                }
                if (!first) {
                    output.append(',');
                }
                first = false;
                quote(output, key);
                output.append(':');
                encodeValue(output, entry.getValue(), depth + 1);
            }
            output.append('}');
        } else if (value instanceof List<?> array) {
            output.append('[');
            for (int position = 0; position < array.size(); position++) {
                if (position > 0) {
                    output.append(',');
                }
                encodeValue(output, array.get(position), depth + 1);
            }
            output.append(']');
        } else {
            throw new IllegalArgumentException("unsupported JSON value type");
        }
    }

    private static void requireFinite(double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("JSON numbers must be finite");
        }
    }

    private static void quote(StringBuilder output, String value) {
        output.append('"');
        for (int position = 0; position < value.length(); position++) {
            char current = value.charAt(position);
            switch (current) {
                case '"' -> output.append("\\\"");
                case '\\' -> output.append("\\\\");
                case '\b' -> output.append("\\b");
                case '\f' -> output.append("\\f");
                case '\n' -> output.append("\\n");
                case '\r' -> output.append("\\r");
                case '\t' -> output.append("\\t");
                default -> {
                    if (current < 0x20) {
                        output.append(String.format("\\u%04x", (int) current));
                    } else {
                        output.append(current);
                    }
                }
            }
        }
        output.append('"');
    }

    private void validateSurrogates(StringBuilder value) throws ProtocolException {
        for (int position = 0; position < value.length(); position++) {
            char current = value.charAt(position);
            if (Character.isHighSurrogate(current)) {
                if (position + 1 >= value.length()
                        || !Character.isLowSurrogate(value.charAt(position + 1))) {
                    throw error("unpaired unicode surrogate");
                }
                position++;
            } else if (Character.isLowSurrogate(current)) {
                throw error("unpaired unicode surrogate");
            }
        }
    }

    private void requireDigitOneToNine() throws ProtocolException {
        char value = peekOrZero();
        if (value < '1' || value > '9') {
            throw error("expected number");
        }
        index++;
    }

    private void requireDigit() throws ProtocolException {
        if (!isDigit(peekOrZero())) {
            throw error("expected digit");
        }
        index++;
    }

    private static boolean isDigit(char value) {
        return value >= '0' && value <= '9';
    }

    private void skipWhitespace() {
        while (index < input.length()) {
            char current = input.charAt(index);
            if (current != ' ' && current != '\n' && current != '\r' && current != '\t') {
                return;
            }
            index++;
        }
    }

    private char peek() throws ProtocolException {
        if (index >= input.length()) {
            throw error("unexpected end of message");
        }
        return input.charAt(index);
    }

    private char peekOrZero() {
        return index < input.length() ? input.charAt(index) : '\0';
    }

    private void expect(char expected) throws ProtocolException {
        if (!take(expected)) {
            throw error("expected '" + expected + "'");
        }
    }

    private boolean take(char expected) {
        if (index < input.length() && input.charAt(index) == expected) {
            index++;
            return true;
        }
        return false;
    }

    private ProtocolException error(String message) {
        return new ProtocolException(message + " at character " + index);
    }
}
