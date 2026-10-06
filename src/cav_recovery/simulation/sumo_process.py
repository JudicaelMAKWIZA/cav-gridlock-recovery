"""Ferme TraCI et le processus SUMO."""


def close_sumo(connection, process, result: dict) -> None:
    """Ferme TraCI puis attend SUMO, avec arrêt de secours si nécessaire."""
    if connection is not None:
        try:
            connection.close(False)
            result["connection_closed"] = True
        except Exception as error:
            result["cleanup_errors"].append(f"Fermeture TraCI impossible : {error}")
    if process is not None:
        # Une erreur d'attente ne signifie pas que SUMO est arrêté.
        # On essaie alors terminate, puis kill si nécessaire.
        for action, phase in (
            (None, "attente normale"),
            (process.terminate, "terminate"),
            (process.kill, "kill"),
        ):
            if action is not None:
                result["forced_process_stop"] = True
                try:
                    action()
                except Exception as error:
                    result["cleanup_errors"].append(f"Arrêt SUMO par {phase} impossible : {error}")
            try:
                process.wait(timeout=5)
                break
            except Exception as error:
                result["cleanup_errors"].append(f"Attente SUMO ({phase}) impossible : {error}")
        try:
            returncode = process.poll()
            result["process_stopped"] = returncode is not None
            result["process_returncode"] = returncode
        except Exception as error:
            result["cleanup_errors"].append(f"Vérification de l'arrêt SUMO impossible : {error}")
        if not result["process_stopped"]:
            result["cleanup_errors"].append("L'arrêt du processus SUMO n'a pas pu être confirmé.")
