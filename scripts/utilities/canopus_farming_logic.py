"""Farmer de la Estrategia Canopus.

Mismo molde que el resto de los farmers (``IFarmer`` + ``run_state_loop`` +
``FarmingFactory``: login, dailies, botón Update de la GUI, detección de
"bot atascado", etc. vienen gratis por herencia), pero SIN capa de
navegación: arranca directamente en ``FIGHTING``, igual que hacía
``EstrategiaCanopus.py`` antes de este cambio. La pelea de Demon King con el
equipo Canopus se entra a mano -no hay una pantalla de dificultad
estandarizada como hard/hell que este farmer pueda navegar-, así que no
existen estados "ir al menú" / "abrir la pelea": el usuario ya tiene que
estar parado en la pelea cuando le da Start.
"""

import threading
from enum import Enum, auto

import utilities.vision_images as vio
from utilities.canopus_fighter import CanopusFighter
from utilities.canopus_fighting_strategies import CanopusCarouselStrategy
from utilities.general_farmer_interface import IFarmer
from utilities.general_fighter_interface import IFighter
from utilities.utilities import capture_window, find_and_click


class States(Enum):
    FIGHTING = auto()
    EXIT_FARMER = auto()


class CanopusFarmer(IFarmer):
    """Farmer de la Estrategia Canopus (Demon King, carrusel de 4 aliados)."""

    # Compartido entre instancias, igual que success_count/total_count en los demás farmers.
    success_count = 0
    total_count = 0

    def __init__(
        self,
        starting_state: States = States.FIGHTING,
        battle_strategy=None,  # No se usa: Canopus siempre corre CanopusCarouselStrategy.
        max_runs="inf",
        do_dailies: bool = False,
        do_daily_pvp: bool = True,
        password: str | None = None,
        **kwargs,
    ):
        super().__init__(do_daily_pvp=do_daily_pvp)

        if password:
            IFarmer.password = password
            print("Stored the account password locally in case we need to log in again.")

        IFarmer.do_dailies = do_dailies

        self.current_state = starting_state
        self.max_runs = float(max_runs)
        if self.max_runs < float("inf"):
            print(f"Vamos a limpiar la pelea {int(self.max_runs)} veces.")

        # Composición: el fighter contiene toda la lógica del carrusel A/B/C
        # (ver utilities/canopus_fighting_strategies.py). Le pasamos el
        # callback para que nos avise cuándo termina cada pelea.
        self.fighter: IFighter = CanopusFighter(
            battle_strategy=CanopusCarouselStrategy,
            callback=self.fight_complete_callback,
        )
        self.fight_thread: threading.Thread | None = None

        # Para el login/dailies (igual que el resto de los farmers).
        IFarmer.daily_farmer.add_complete_callback(self.dailies_complete_callback)

    def fighting_state(self):
        """Arranca (o mantiene) la pelea. Sin navegación previa: se asume que
        ya estás parado en la pelea de Demon King con el equipo Canopus.
        """
        screenshot, window_location = capture_window()

        if self.check_for_dailies():
            return
        self.maybe_reset_daily_checkin_flag()

        # Saltar animaciones de inicio de pelea, si las hay.
        find_and_click(vio.skip, screenshot, window_location)

        if (self.fight_thread is None or not self.fight_thread.is_alive()) and self.current_state == States.FIGHTING:
            print("Estrategia Canopus: pelea iniciada.")
            self.fighter.prepare_for_new_fight()
            self.fight_thread = threading.Thread(
                target=self.fighter.run,
                name="CanopusFighterThread",
                daemon=True,
            )
            self.fight_thread.start()

    def fight_complete_callback(self, victory: bool = False, **kwargs):
        """Llamado por el CanopusFighter cuando termina una pelea (ganada o perdida)."""
        with IFarmer._lock:
            CanopusFarmer.total_count += 1
            if victory:
                CanopusFarmer.success_count += 1
                print("¡PELEA GANADA!")
                print("[CLEAR]")
            else:
                phase = kwargs.get("phase")
                print(f"Pelea perdida{f' en fase {phase}' if phase is not None else ''}...")
                print("[LOSS]")

            print(f"Resultados de la sesión: {CanopusFarmer.success_count}/{CanopusFarmer.total_count} victorias.")

            if CanopusFarmer.success_count >= self.max_runs:
                print("Alcanzamos el máximo de victorias pedido; cerrando el farmer.")
                self.current_state = States.EXIT_FARMER
                return

            # Volver a FIGHTING: fighting_state() lanza la siguiente pelea sola.
            self.current_state = States.FIGHTING

    def dailies_complete_callback(self):
        """La dailies thread nos avisa que terminó; volvemos a la Estrategia Canopus."""
        with IFarmer._lock:
            print("¡Dailies completadas! Volviendo a la Estrategia Canopus.")
            IFarmer.dailies_thread = None
            self.current_state = States.FIGHTING

    def exit_message(self):
        super().exit_message()
        percent = (
            (CanopusFarmer.success_count / CanopusFarmer.total_count) * 100 if CanopusFarmer.total_count > 0 else 0
        )
        print(
            f"Estrategia Canopus: {CanopusFarmer.success_count}/{CanopusFarmer.total_count} "
            f"victorias ({percent:.2f}%)."
        )

    def run(self):
        print("Estrategia Canopus (carrusel) iniciada. Entra a la pelea; actuaré cuando sea tu turno.")
        self.run_state_loop(
            {
                States.FIGHTING: self.fighting_state,
                States.EXIT_FARMER: self.exit_farmer_state,
            },
            login_return_state=States.FIGHTING,
            sleep_seconds=0.6,
        )
