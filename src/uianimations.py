from PyQt6 import QtCore, QtWidgets

class UIAnimations:
    @staticmethod
    def fadeInUp(widget):
        final_pos=widget.pos()
        start_pos=final_pos+QtCore.QPoint(0,30)
        widget.move(start_pos)

        effect=QtWidgets.QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        effect.setOpacity(0)

        fade=QtCore.QPropertyAnimation(effect,b"opacity")
        fade.setDuration(700)
        fade.setStartValue(0)
        fade.setEndValue(1)
        fade.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

        move_anim=QtCore.QPropertyAnimation(widget,b"pos")
        move_anim.setDuration(700)
        move_anim.setStartValue(start_pos)
        move_anim.setEndValue(final_pos)
        move_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        group=QtCore.QParallelAnimationGroup(widget)
        group.addAnimation(fade)
        group.addAnimation(move_anim)
        group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    