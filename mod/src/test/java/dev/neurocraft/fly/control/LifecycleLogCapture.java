package dev.neurocraft.fly.control;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;
import org.apache.logging.log4j.core.layout.PatternLayout;

/** Actual logger capture; diagnostic assertions do not replace transport callbacks. */
final class LifecycleLogCapture extends AbstractAppender implements AutoCloseable {
    private final Logger logger;
    private final Level previousLevel;
    private final List<LogEvent> events = new CopyOnWriteArrayList<>();

    LifecycleLogCapture(String loggerName) {
        super("lifecycle-test-capture", null, PatternLayout.createDefaultLayout(), false, Property.EMPTY_ARRAY);
        logger = (Logger) LogManager.getLogger(loggerName);
        previousLevel = logger.getLevel();
        start();
        logger.addAppender(this);
        logger.setLevel(Level.WARN);
    }

    @Override public void append(LogEvent event) { events.add(event.toImmutable()); }

    List<String> messages() {
        return events.stream().map(event -> {
            if (event.getLevel() != Level.WARN) throw new AssertionError("lifecycle must be visible at WARN");
            if (event.getThrown() != null) throw new AssertionError("diagnostics must not log exception payloads");
            return event.getMessage().getFormattedMessage();
        }).toList();
    }

    List<String> stages() {
        return messages().stream().map(message -> message.split(" stage=", 2)[1].split(" ", 2)[0]).toList();
    }

    @Override public void close() {
        logger.removeAppender(this);
        logger.setLevel(previousLevel);
        stop();
    }
}
