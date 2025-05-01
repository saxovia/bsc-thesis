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
                label._typing_timer.deleteLater()
            except:
                pass
            label._typing_timer = None

        typing_timer = QtCore.QTimer(parent)
        label._typing_timer = typing_timer
        typing_index = 0

        def update_typing():
            nonlocal typing_index
            if typing_index < len(text):
                label.setText(text[:typing_index + 5])
                typing_index += 5
            else:
                label.setText(text)
                try:
                    typing_timer.stop()
                    typing_timer.deleteLater()
                except:
                    pass
                label._typing_timer = None

        def start_typing():
            if hasattr(label, '_typing_timer') and label._typing_timer == typing_timer:
                try:
                    typing_timer.timeout.connect(update_typing)
                    typing_timer.start(interval)
                except RuntimeError:
                    pass

        label.setText(" ")
        QtCore.QTimer.singleShot(500, start_typing)