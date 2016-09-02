use sqlite::{Connection, Statement, State};
use std::mem;

use streamer::{Config, Result};
use streamer::platform::{self, Platform, Profile};
use streamer::system::{Event, EventKind};

pub struct Output {
    element_id: usize,
    quantity: Quantity,
    batch_size: usize,
    accumulator: f64,
    position: usize,
    connection: Connection,
    events: Statement<'static>,
    profile: Statement<'static>,
}

#[derive(Clone, Copy)]
enum Quantity {
    Power,
    Temperature,
}

impl Output {
    pub fn new(platform: &platform::Thermal, config: &Config) -> Result<Self> {
        use sql::prelude::*;

        let element_id = *some!(config.get::<i64>("element_id"), "an element ID is required");
        let element_id = match element_id {
            value if value >= 0 && value < platform.elements().len() as i64 => value as usize,
            _ => raise!("the element ID is invalid"),
        };
        let quantity = some!(config.get::<String>("quantity"), "an output quantity is required");
        let quantity = match &quantity[..] {
            "power" => Quantity::Power,
            "temperature" => Quantity::Temperature,
            _ => raise!("the output quantity is invalid"),
        };
        let batch_size = *config.get::<i64>("batch_size").unwrap_or(&1);
        let batch_size = match batch_size {
            value if value > 0 => value as usize,
            _ => raise!("the batch size is invalid"),
        };
        let connection = ok!(Connection::open(path!(@unchecked config,
                                                    "an output file is required")));
        ok!(connection.execute("
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
        "));
        let events = {
            ok!(connection.execute(
                ok!(create_table("events").if_not_exists().columns(&[
                    "time".integer().not_null(), "value".integer().not_null(),
                ]).compile())
            ));
            ok!(connection.execute(ok!(delete_from("events").compile())));
            let statement = ok!(connection.prepare(
                ok!(insert_into("events").columns(&["time", "value"]).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        let profile = {
            ok!(connection.execute(
                ok!(create_table("profile").if_not_exists().columns(&[
                    "time".integer().not_null(), "value".float().not_null(),
                ]).compile())
            ));
            ok!(connection.execute(ok!(delete_from("profile").compile())));
            let statement = ok!(connection.prepare(
                ok!(insert_into("profile").columns(&["time", "value"]).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        Ok(Output {
            element_id: element_id,
            events: events,
            profile: profile,
            accumulator: 0.0,
            position: 0,
            quantity: quantity,
            batch_size: batch_size,
            connection: connection,
        })
    }

    pub fn next(&mut self, event: &Event, profiles: &(Profile, Profile)) -> Result<()> {
        ok!(self.connection.execute("BEGIN TRANSACTION"));
        ok!(self.process_profile(profiles));
        ok!(self.process_event(event));
        ok!(self.connection.execute("END TRANSACTION"));
        Ok(())
    }

    fn process_event(&mut self, event: &Event) -> Result<()> {
        let (value, mapping) = match &event.kind {
            &EventKind::Start(_, ref mapping) => (0, mapping),
            &EventKind::Finish(_, ref mapping) => (1, mapping),
            _ => return Ok(()),
        };
        for &(_, i) in mapping {
            if i == self.element_id {
                try!(self.write_event(value));
                break;
            }
        }
        Ok(())
    }

    fn process_profile(&mut self, profiles: &(Profile, Profile)) -> Result<()> {
        let &Profile { element_count, step_count, ref data, .. } = match self.quantity {
            Quantity::Power => &profiles.0,
            Quantity::Temperature => &profiles.1,
        };
        for i in 0..step_count {
            self.accumulator += data[i * element_count + self.element_id];
            self.position += 1;
            if self.position % self.batch_size == 0 {
                let value = self.accumulator / self.batch_size as f64;
                try!(self.write_profile(value));
                self.accumulator = 0.0;
            }
        }
        Ok(())
    }

    fn write_event(&mut self, value: i64) -> Result<()> {
        let left_count = self.position % self.batch_size;
        if left_count > 0 {
            self.position += self.batch_size - left_count;
            let value = self.accumulator / left_count as f64;
            try!(self.write_profile(value));
            self.accumulator = 0.0;
        }
        let statement = &mut self.events;
        ok!(statement.reset());
        ok!(statement.bind(1, (self.position / self.batch_size) as i64));
        ok!(statement.bind(2, value));
        if State::Done != ok!(statement.next()) {
            raise!("failed to write into the database");
        }
        Ok(())
    }

    fn write_profile(&mut self, value: f64) -> Result<()> {
        let statement = &mut self.profile;
        ok!(statement.reset());
        ok!(statement.bind(1, (self.position / self.batch_size) as i64));
        ok!(statement.bind(2, value));
        if State::Done != ok!(statement.next()) {
            raise!("failed to write into the database");
        }
        Ok(())
    }
}
