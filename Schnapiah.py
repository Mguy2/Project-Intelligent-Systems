import os.path
import random
import threading
import numpy as np
import nashpy as nash
import logging
import time
from typing import Optional

import click
import schnapsen
from schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot
from schnapsen.bots.example_bot import ExampleBot
from schnapsen.game import (Bot, Move, PlayerPerspective, SchnapsenGamePlayEngine, Trump_Exchange, SchnapsenTrickScorer)
from schnapsen.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine
from schnapsen.bots.rdeep import RdeepBot
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from schnapsen.game import GamePhase
from typing import Iterable, List, Optional, Tuple, Union, cast, Any
#from ..deck import CardCollection, OrderedCardCollection, Card, Rank, Suit
import itertools


class Schnapiah(Bot):
    """Schnapiah Bot.

    Args:
        Bot (Bot): Schnapiah inherits from Class bot in order to get notified of game events.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.history = dict()
        self.turn = 0
        

    def __str__(self) -> str:
        return f"Schnapsen Overlord"


    def trump_exchange(self) -> Move:
        for moves in self.moves:
            if moves.is_trump_exchange():
                return moves
        return False
    
    
    def marriage(self) -> Move:
        for moves in self.moves:
            if moves.is_marriage():
                return moves
        return False
    

    def generate_matrice(self) -> List[List[float]]:
        matrice = list()
        matrice_readable = list()
        matrice_header = list()
        for cards_in_hand in self.moves:
           #print("MOVES", self.moves)
            row = list()
            for cards_opp in self.possible_cards:
                points_my_card = float(SchnapsenTrickScorer().rank_to_points(cards_in_hand.cards[0].rank))
                points_opp_card = float(SchnapsenTrickScorer().rank_to_points(cards_opp.rank))
                points = points_my_card + points_opp_card
                # Win or lose a move?
                if points_my_card < points_opp_card and self.trump_suit == cards_in_hand.cards[0].suit and self.trump_suit != cards_opp.suit:
                    # Trump for us
                    points -= (self.highest_card_opponent[1] - points_opp_card)
                if points_my_card < points_opp_card and self.trump_suit != cards_in_hand.cards[0].suit and self.trump_suit != cards_opp.suit:
                    # Opponent wins
                    points *= -1
                    points += (self.highest_card_hand[1] - points_my_card)
                if points_my_card < points_opp_card and self.trump_suit == cards_in_hand.cards[0].suit and self.trump_suit == cards_opp.suit:
                    # Opponent wins
                    points *= -1
                    points += (self.highest_card_hand[1] - points_my_card)
                if points_my_card < points_opp_card and self.trump_suit != cards_in_hand.cards[0].suit and self.trump_suit == cards_opp.suit:
                    # Opponent wins
                    points *= -1
                    points += (self.highest_card_hand[1] - points_my_card)
                if points_my_card > points_opp_card and self.trump_suit != cards_in_hand.cards[0].suit and self.trump_suit == cards_opp.suit:
                    # Opponent wins
                    points *= -1
                    points += (self.highest_card_hand[1] - points_my_card)
                if points_my_card > points_opp_card and self.trump_suit == cards_in_hand.cards[0].suit and self.trump_suit != cards_opp.suit:
                    # We Win
                    points -= (self.highest_card_opponent[1] - points_opp_card)
                if points_my_card > points_opp_card and self.trump_suit == cards_in_hand.cards[0].suit and self.trump_suit == cards_opp.suit:
                    # We Win 
                    points -= (self.highest_card_opponent[1] - points_opp_card)                 
                if points_my_card > points_opp_card and self.trump_suit != cards_in_hand.cards[0].suit and self.trump_suit != cards_opp.suit:    
                    # We Win    
                    points -= (self.highest_card_opponent[1] - points_opp_card)
                if not self.leader and points_my_card == points_opp_card:
                    points *= -1
                elif cards_in_hand.cards[0].suit == self.trump_suit and cards_opp.suit == self.trump_suit:
                    if points_my_card < points_opp_card:
                        points *= -1
                    else:
                        points -= (self.highest_card_opponent[1])
                row.append(points)
            matrice.append(row)
            matrice_readable.append({cards_in_hand.cards[0]: row})
        for cards_opp in self.possible_cards:
            matrice_header.append(cards_opp)
        self.matrix = matrice
        #print(nash.Game(self.matrix),"\n\n")
        #print(matrice_header, "\n", np.array(matrice_readable), "\n", "trump suit:", self.trump_suit, "==== Leader:", self.leader, "==== Phase:", self.phase, "\n\n")

    
    def calculate_probability(self) -> None:
        known_cards = dict()
        for cards in self.cards_opponent_spec:
            known_cards[cards] = known_cards.get(cards, 1)
        #print(known_cards)


    def generate_possible_cards_opponent(self, state: PlayerPerspective, leader_move: Optional[Move]) -> None:
        full_deck = state.get_engine().deck_generator.get_initial_deck().get_cards()
        possible = list()
        seen = list()
        if not leader_move:
            if self.trump_card:
                seen.append(str(self.trump_card.suit) + " " + str(self.trump_card.rank))
            for cards in self.cards_won:
                seen.append(str(cards.suit) + " " + str(cards.rank))
            for cards in self.cards_won_opponent:
                seen.append(str(cards.suit) + " " + str(cards.rank))
            for cards in self.cards_hand:
                seen.append(str(cards.suit) + " " + str(cards.rank))
            for cards in full_deck:
                cards_str = str(cards.suit) + " " + str(cards.rank)
                if not cards_str in seen:
                    possible.append(cards)
        else:
            possible.append(leader_move.cards[0])
        self.calculate_probability()
        self.possible_cards = possible
        #print("Phase", self.phase, "\n\nFULL DECK", full_deck, "\n\nPossible", possible, len(possible))
        

    def find_conditions(self, state: PlayerPerspective) -> None:
        # Am I currently playing as the leader or follower?
        if state.am_i_leader():
            self.leader = True
        else:
            self.leader = False
        # What phase of the game are we currently in?
        if state.get_phase() == GamePhase.ONE:
            self.phase = 1
        else:
            self.phase = 2
        # Get my score and opponent score.
        self.score = state.get_my_score()
        self.score_opponent = state.get_opponent_score()
        # Get the suit of the trump
        self.trump_suit = state.get_trump_suit()
        # Get valid moves
        self.moves = state.valid_moves()
        # Check for trump exchange
        self.trump_exch_move = self.trump_exchange()
        # Check for marriage
        self.marriage_move = self.marriage()
        

    def counting_cards(self, state: PlayerPerspective, leader_move: Optional[Move]) -> None:
        # Get size of the tallon
        self.cards_tallon_size = state.get_talon_size()
        # Get the cards we won and the opponent won
        self.cards_won = state.get_won_cards().get_cards()
        self.cards_won_opponent = state.get_opponent_won_cards().get_cards()
        # Get the cards known to be in the hand of the opponent.
        self.cards_opponent_spec = state.get_known_cards_of_opponent_hand()
        # List of all cards seen before:
        self.cards_seen = state.seen_cards(leader_move)
        self.cards_seen_nice = self.cards_seen.get_cards()
        # Our own hand.
        self.cards_hand = state.get_hand().get_cards()
        # Get the Trump Card
        self.trump_card = state.get_trump_card()
        # Get Game History
        self.historic_moves = state.get_game_history()
        # If phase is two, get all the cards in the opponent's hand.
        if state.get_phase() == GamePhase.TWO:
            self.cards_hand_opponent = state.get_opponent_hand_in_phase_two()


    def opportunity_cost(self) -> None:
        highest_card_hand = ["", 0]
        highest_card_opponent = ["", 0]
        for moves in self.moves:
            if SchnapsenTrickScorer().rank_to_points(moves.cards[0].rank) > highest_card_hand[1]:
                highest_card_hand = [moves.cards[0], SchnapsenTrickScorer().rank_to_points(moves.cards[0].rank)]
        for cards in self.possible_cards:
            if SchnapsenTrickScorer().rank_to_points(cards.rank) > highest_card_opponent[1]:
                highest_card_opponent = [cards, SchnapsenTrickScorer().rank_to_points(cards.rank)]
        self.highest_card_hand = highest_card_hand
        self.highest_card_opponent = highest_card_opponent
        
    
    def document_moves(self, state: PlayerPerspective, leader_move: Optional[Move], move_made: Move) -> None:
        self.turn += 1
        if leader_move:
            self.history[self.turn] = {"Opponent": leader_move.cards[0], "Self": move_made, "Leader": self.leader}
        else:
            self.history[self.turn] = {"Opponent": False, "Self": move_made, "Leader": self.leader}
        for opponent_moves in self.history.values():
            if not opponent_moves.get("Opponent"):
                pass
                #print(opponent_moves.get("Opponent"))
            #print(opponent_moves)
        # for moves in self.historic_moves:
        #     if moves[1]:
        #         print(moves[1].cards,"\n")
        #         print(state.seen_cards(leader_move))
    
    
    def best_overall_model(self) -> Move:
        best = 0
        move = 0
        if self.trump_exch_move:
            return self.trump_exch_move
        if self.marriage_move:
            return self.marriage_move
        for index, rows in enumerate(self.matrix):
            if sum(rows) > best:
                best = sum(rows)
                move = index
        return self.moves[move]
    
    
    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # Create two threads to utilize more processing power, decrease wall-clock time.
        thread_find_conditions = threading.Thread(None, self.find_conditions(state), "Find Conditions")
        thread_find_conditions.start()
        thread_counting_cards = threading.Thread(None, self.counting_cards(state, leader_move), "Counting Cards")
        thread_counting_cards.start()
        thread_counting_cards.join()
        thread_find_conditions.join()
        self.generate_possible_cards_opponent(state, leader_move)
        self.opportunity_cost()
        self.generate_matrice()
        #if not self.leader:
            #print("Leader Move:", leader_move)
        #print("Move made:", self.moves[move], "\nTrump Suit:", self.trump_suit, "\nMy points:", self.score, "OPP points:", self.score_opponent, "\n\n")
        make_move = self.best_overall_model()
        self.document_moves(state, leader_move, make_move)
        return make_move

