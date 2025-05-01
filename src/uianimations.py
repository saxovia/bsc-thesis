from PyQt6 import QtCore, QtWidgets

class UIAnimations:
    @staticmethod
    def fade_in_up(widget):
        if widget is None:
            return None
    
        final_pos = widget.pos()
        start_pos = final_pos + QtCore.QPoint(0, 30)
        widget.move(start_pos)

        effect = QtWidgets.QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        effect.setOpacity(0)

        fade = QtCore.QPropertyAnimation(effect, b"opacity")
        fade.setDuration(200)
        fade.setStartValue(0)
        fade.setEndValue(1)
        fade.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

        move_anim = QtCore.QPropertyAnimation(widget, b"pos")
        move_anim.setDuration(200)
        move_anim.setStartValue(start_pos)
        move_anim.setEndValue(final_pos)
        move_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        def on_animation_finished():
            widget.move(final_pos)  # Ensure the widget is at the final position
            widget.updateGeometry()  # Update the layout to reflect the new position

        group = QtCore.QParallelAnimationGroup(widget)
        group.addAnimation(fade)
        group.addAnimation(move_anim)
        group.finished.connect(on_animation_finished)  # Connect the finished signal
        group.start(QtCore.QAbstractAnimation.DeletionPolicy.KeepWhenStopped)

        return effect
    
    @staticmethod
    def type_text_effect(label, text, parent, interval=10):
        if hasattr(label, '_typing_timer'):
            try:
                label._typing_timer.stop()
                label._typing_timer.timeout.disconnect()
            except:
                pass
            del label._typing_timer

        # Create new timer
        label._typing_timer = QtCore.QTimer(parent)
        typing_timer = label._typing_timer
        typing_index = 0

        def update_typing():
            nonlocal typing_index
            if typing_index < len(text):
                label.setText(text[:typing_index + 5])
                typing_index += 5
            else:
                label.setText(text)
                typing_timer.stop()
                # Clean up the timer when done
                try:
                    typing_timer.timeout.disconnect()
                except:
                    pass

        def start_typing():
            typing_timer.timeout.connect(update_typing)
            typing_timer.start(interval)

        label.setText(" ")
        QtCore.QTimer.singleShot(500, start_typing)